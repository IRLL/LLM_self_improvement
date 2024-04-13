import transformers
import torch,tqdm
import os,json,sys

from utils import log_method, ClearCache,load_tokenizer,load_model_with_adapters,split_into_batches

def inference(model, tokenizer, batch_input_text):
    input_ids = tokenizer(batch_input_text, return_tensors="pt", max_length=2048,padding=True, truncation=True).to('cuda:0')
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'], 
            do_sample=True, 
            use_cache=True, 
            num_return_sequences=1,
            max_new_tokens=20,
            attention_mask=input_ids['attention_mask'] ,
            pad_token_id=tokenizer.pad_token_id
        )
    #generated_texts = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]

    res = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]
    del input_ids
    return res

@log_method
def eval_squad():
    arguments = json.loads(sys.argv[1])

    iteration = int(arguments['cur_iteration'])
    transformed_squad_eval_set_path = arguments['transformed_squad_eval_set_path']
    original_squad_eval_set_path = arguments['original_squad_eval_set_path']
    squad_response_gen_file = arguments['squad_response_gen_file']
    squad_eval_result_path = arguments['squad_eval_result_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']   
    inference_batch_size=int(arguments['inference_batch_size'])
     
    with ClearCache():
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        # pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     torch_dtype=torch.float16,
        #     max_new_tokens=20)

        # pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
        with open(transformed_squad_eval_set_path) as obj:
            file = json.loads(obj.read())

        # all_data = file["data"]#["paragraphs"]["qas"]
        res_dict = {}

        prompt = """Write a response that appropriately completes answer the question, follow the examples. You should answer 'no answer found' if you cannot find the answer from the context.

        ### Example 1:
        Context: 
        A problem is regarded as inherently difficult if its solution requires significant resources, whatever the algorithm used. The theory formalizes this intuition, by introducing mathematical models of computation to study these problems and quantifying the amount of resources needed to solve them, such as time and storage.

        Question:
        What method is used to intuitively assess or quantify the amount of resources required to solve a computational problem?

        Answer:
        mathematical models of computation

        ### Example 2:
        Context: 
        Under the terms of the Scotland Act 1978, an elected assembly would be set up in Edinburgh provided that the majority of the Scottish electorate voted for it in a referendum to be held on 1 March 1979 that represented at least 40% of the total electorate. The 1979 Scottish devolution referendum to establish a devolved Scottish Assembly failed.

        Question:
        President Wilson committed his government to what in 1974?

        Answer:
        no answer found

        ### Task:
        Context:
        {context}

        Question:
        {question}

        Answer:"""

        batches = split_into_batches(file, inference_batch_size)

        for each_batch in tqdm.tqdm(batches,desc="squad_eval"):
            full_prompt_list = [prompt.format(question=each_row['question'], context=each_row['context']) for each_row in each_batch]

            res = inference(model, tokenizer, full_prompt_list)
            for idx, each_output in enumerate(res):
                            
                output_text = each_output[len(full_prompt_list[idx]):]
                truncated_result = output_text.strip()

                answer = truncated_result

                if "no answer" in truncated_result.lower() or len(answer)<=3:
                    answer = ""

                res_dict[each_batch[idx]['id']] = answer

        with open(squad_response_gen_file,'w') as obj:
            obj.write(json.dumps(res_dict))

        del res_dict, file
        os.system(f"python scripts/official_squad_eval.py --data_file='{original_squad_eval_set_path}' --pred_file='{squad_response_gen_file}' --out-file={squad_eval_result_path}")

        with open(squad_eval_result_path) as obj:
            squad_result = json.loads(obj.read())

        print(f"squad result for iter {iteration}",squad_result)

    