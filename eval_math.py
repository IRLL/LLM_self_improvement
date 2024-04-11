import transformers
import torch
import os,tqdm,json,time
from utils import *

@log_method
def eval_gsm8k(model, tokenizer, gsm8k_eval_path, gsm8k_eval_result_path):
    with ClearCache():
        model.eval()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=150,
            do_sample=True


        )

        with open(gsm8k_eval_path) as obj:
            gsm8k_data = json.loads(obj.read())
        
        gsm8k_data = gsm8k_data[:500]

        prompt = """Write a response that appropriately completes answer the math question, follow the examples. You must end your response with "The answer is []".
        Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n
        Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n
        Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\n
        Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n
        Q: {question}\nA: """

        predictions = []

        for item in tqdm.tqdm(gsm8k_data,desc="math_eval"): 
            full_prompt = prompt.format(question=item['question'])
            result = pipeline(full_prompt)
            truncated = result[0]['generated_text'][len(full_prompt):].strip().split('\n')[0]
            predicted_answer = truncated[truncated.find('The answer is')+len("The answer is"):]
            label = item["answer"][item["answer"].find('#### ')+5:]

            if label in predicted_answer:
                predictions.append(1)

            else: 
                predictions.append(0)



        acc = sum(predictions)/len(predictions)
        print("accuracy for math eval",acc)
        del pipeline, predictions, gsm8k_data
        with open(gsm8k_eval_result_path,'w') as obj:
            obj.write(json.dumps({"acc":acc}))


    
    return acc

# model = load_model("/home/qianxi/scratch/laffi/models/7b", four_bit_quant=True, adapter_path=None)
# tokenizer = load_tokenizer("/home/qianxi/scratch/laffi/models/7b")
# gsm8k_eval_path = "/home/qianxi/scratch/laffi/datasets/GSM8K/grade_school_math/data/test.json"
# gsm8k_eval_result_path = "/home/qianxi/scratch/laffi/datasets/GSM8K/grade_school_math/data/test.json"
# eval_gsm8k(model, tokenizer, gsm8k_eval_path, gsm8k_eval_result_path=None)