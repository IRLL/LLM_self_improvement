import transformers
import torch
import os,tqdm,json,sys

from utils import calculate_classification_metrics,log_method,ClearCache,load_model_with_adapters,load_tokenizer,split_into_batches

def inference(model, tokenizer, batch_input_text):
    input_ids = tokenizer(batch_input_text, return_tensors="pt",max_length=2048, padding=True, truncation=True).to('cuda:0')
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'], 
            do_sample=True, 
            use_cache=True, 
            num_return_sequences=1,
            max_new_tokens=10,
            attention_mask=input_ids['attention_mask'] ,
            pad_token_id=tokenizer.pad_token_id
        )
    #generated_texts = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]

    res = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]
    del input_ids
    return res

@log_method
def eval_boolq():
    arguments = json.loads(sys.argv[1])

    iteration = int(arguments['cur_iteration'])
    boolq_eval_result_path = arguments['boolq_eval_result_path']
    boolq_eval_path = arguments['boolq_eval_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    inference_batch_size=int(arguments['inference_batch_size'])


    with ClearCache():

        model = load_model_with_adapters(iteration, adapters_path, model_path)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        with open(boolq_eval_path) as obj:
            boolq_data = json.loads(obj.read())

        #print(len(boolq_data))
        boolq_data =boolq_data[:1000]

        prompt = """Write a response that appropriately completes answer the question, follow the examples. Your answer should be "True" or "False".

        ### Example 1:
        Passage: 
        The Vampire Diaries, an American supernatural drama, was renewed for an eighth season by The CW on March 11, 2016. On July 23, 2016, the CW announced that the upcoming season would be the series' last and would consist of 16 episodes. The season premiered on October 21, 2016 and concluded on March 10, 2017.

        Question:
        will there be a season 8 of vampire diaries?

        Answer:
        True

        ### Example 2:
        Passage: 
        This is the list of U.S. states that have participated in the Little League World Series. As of the 2018 LLWS, eight states had never reached the LLWS: Alaska, Colorado, Kansas, North Dakota, Utah, Vermont, Wisconsin, and Wyoming; additionally, the District of Columbia has never reached the LLWS.

        Question:
        has wisconsin ever been in the little league world series?

        Answer:
        False

        ### Task:
        Passage:
        {passage}

        Question:
        {question}

        Answer:
        """


        """
        batches = split_into_batches(texts, inference_batch_size)
                for each_batch in tqdm.tqdm(batches):


                    res = inference(model, tokenizer, each_batch)

                    for idx, each_output in enumerate(res):
                            
                        output_text = each_output[len(each_batch[idx]):]
                        truncated_result = output_text.strip()

                        result.append(truncated_result)

        """
        batches = split_into_batches(boolq_data, inference_batch_size)
        predictions = []
        labels = []
        for each_batch in tqdm.tqdm(batches):

            full_prompt_list = [prompt.format(question=item['question'], passage=item['passage']) for item in each_batch]
            res = inference(model, tokenizer, full_prompt_list)
            for idx, each_output in enumerate(res):
                            
                output_text = each_output[len(full_prompt_list[idx]):]
                truncated_result = output_text.strip()

                if "false" in truncated_result or "False" in truncated_result:
                    predictions.append(0)
                else:
                    predictions.append(1)
                labels.append(each_batch[idx]["answer"])


        metrics = calculate_classification_metrics(predictions, labels)
        print("boolq metrics",metrics)
        with open(boolq_eval_result_path,'w') as obj:
            obj.write(json.dumps(metrics))
        
        del labels,predictions,boolq_data



eval_boolq()