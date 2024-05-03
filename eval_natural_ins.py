import transformers
import torch
import os,tqdm,json,sys
from torchmetrics.text.rouge import ROUGEScore
from utils import log_method,ClearCache,load_model_with_adapters,load_tokenizer,split_into_batches

def inference(model, tokenizer, batch_input_text):
    input_ids = tokenizer(batch_input_text, return_tensors="pt",max_length=2048, padding=True, truncation=True).to('cuda:0')
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'], 
            do_sample=True, 
            use_cache=True, 
            num_return_sequences=1,
            max_new_tokens=100,
            attention_mask=input_ids['attention_mask'] ,
            pad_token_id=tokenizer.pad_token_id
        )
    #generated_texts = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]

    res = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]
    del input_ids
    return res

@log_method
def eval_natural_ins():
    #/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_eval_converted.json
    arguments = json.loads(sys.argv[1])

    iteration = int(arguments['cur_iteration'])
    natural_ins_eval_result_path = arguments['natural_ins_eval_result_path']
    natural_ins_eval_path = arguments['natural_ins_eval_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    inference_batch_size=int(arguments['inference_batch_size'])

    rouge = ROUGEScore()
    with ClearCache():

        model = load_model_with_adapters(iteration, adapters_path, model_path)
        #model = load_model(model_path,True)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        with open(natural_ins_eval_path) as obj:
            natural_ins_data = json.loads(obj.read())
            if len(natural_ins_data) > 1000:
                natural_ins_data = natural_ins_data[:1000]

        labels = [item['label'] for item in natural_ins_data]
        batches = split_into_batches(natural_ins_data, inference_batch_size)
        predictions = []

        for each_batch in tqdm.tqdm(batches):

            full_prompt_list = [item['input'] for item in each_batch]

            res = inference(model, tokenizer, full_prompt_list)
            for idx, each_output in enumerate(res):
                            
                output_text = each_output[len(full_prompt_list[idx]):].strip()

                predictions.append(output_text)


        metrics = rouge(predictions, labels)
        metrics = {k: v.item() for k, v in metrics.items()}

        print("natural_ins metrics",metrics)
        with open(natural_ins_eval_result_path,'w') as obj:
            obj.write(json.dumps(metrics))
        
        del labels,predictions,natural_ins_data



eval_natural_ins()