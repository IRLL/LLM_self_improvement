import transformers
import torch,sys
import os,tqdm,json,time
from utils import *


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
def eval_gsm8k():

    arguments = json.loads(sys.argv[1])

    iteration = int(arguments['cur_iteration'])
    gsm8k_eval_result_path = arguments['gsm8k_eval_result_path']
    gsm8k_eval_path = arguments['gsm8k_eval_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    inference_batch_size=int(arguments['inference_batch_size'])

    with ClearCache():

        model = load_model_with_adapters(iteration, adapters_path, model_path)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        with open(gsm8k_eval_path) as obj:
            gsm8k_data = json.loads(obj.read())
        
        gsm8k_data = gsm8k_data[:4000]

        prompt = """Write a response that appropriately completes answer the math question, follow the examples. You must end your response with "The answer is []".
        Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n
        Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n
        Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n
        Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n
        Q: {question}\nA: """

        predictions = []
        batches = split_into_batches(gsm8k_data, inference_batch_size)

        for each_batch in tqdm.tqdm(batches,desc="math_eval"):
            full_prompt_list = [prompt.format(question=item['question']) for item in each_batch]
            res = inference(model, tokenizer, full_prompt_list)

            for idx, each_output in enumerate(res):

                truncated = each_output[len(full_prompt_list[idx]):].strip().split('\n')[0]
                predicted_answer = truncated[truncated.find('The answer is')+len("The answer is"):]
                label = each_batch[idx]["answer"][each_batch[idx]["answer"].find('#### ')+5:]

                if label in predicted_answer:
                    predictions.append(1)

                else: 
                    predictions.append(0)



        acc = sum(predictions)/len(predictions)
        print("accuracy for math eval",acc)
        del predictions, gsm8k_data
        with open(gsm8k_eval_result_path,'w') as obj:
            obj.write(json.dumps({"acc":acc}))

eval_gsm8k()