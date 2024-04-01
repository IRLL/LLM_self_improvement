import transformers
import torch
import os,tqdm,json,time

import random,copy

from utils import log_method

def construct_answer_prompts(base_dataset_path,
                             per_task_data_row_amount,
                             pos_example_amount=None,
                             neg_example_amount=None):
    """
    The goal of this function is to compose the prompts for answer generation.

    for each task:
        1. Accepts (1)the dataset path (2) task examples,
        sample data rows form the full dataset and from different
        tasks.
        2. Compose full prompts for answer generation
        3. Return a dictionary and write a dict to a json file:


    """

    def compose_examples(task_examples, pos_example_amount, negative_example_amount):
        prompt = ""
        
        examples = []
        if pos_example_amount:
            pos_example = task_examples["Positive Examples"][:pos_example_amount]
            examples += pos_example
            for each_example in pos_example:
                prompt += f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]} So the answer is: {each_example["output"]}\n\n"""
            
        if negative_example_amount:
            neg_example = task_examples["Negative Examples"][:negative_example_amount]
            examples += neg_example

            if '-' not in neg_example:
                for each_example in neg_example:
                    prompt +=  f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]} So the answer is: {each_example["output"]}\n\n"""


        return prompt, examples

    
    dataset_dict = {}
    all_prompts = []
    for idx, each_json in enumerate(os.listdir(base_dataset_path)):
        if ".json" in each_json:
            per_task_prompt_list = []
            full_path = os.path.join(base_dataset_path, each_json)

            with open(full_path) as obj:
                file = json.loads(obj.read())
            
            per_task_dict = copy.deepcopy(file)

            full_length = per_task_data_row_amount
            if full_length > len(file["Instances"]):
                full_length = len(file["Instances"])
            instances = file["Instances"][:full_length]

            per_task_dict["Instances"] = instances

            example_prompt, per_task_examples = compose_examples(file["Examples"],
                                                                pos_example_amount,
                                                                neg_example_amount)


            # Compose context.
            instruction = f"""{file["Definition"]} {file["Emphasis & Caution"]}\n\n"""

            context = f"""Please refer to the instruction and task information and give your answers. You need to follow the examples we provided."""
            
            # Compose full_prompt for each instance.
            for idx,instance in enumerate(instances):
                full_prompt = f"""### Context:\n{context}\n\n### Instruction:\n{instruction}### Examples:\n{example_prompt}\n\n### Task:\n{instance['input']}\n\n### Answer:\n"""
                per_task_prompt_list.append(full_prompt)
                

            per_task_dict["Answer Prediction Prompt Dataset"] = per_task_prompt_list
            all_prompts += per_task_dict["Answer Prediction Prompt Dataset"]

            dataset_dict[each_json] = per_task_dict
            del per_task_dict

    dataset_dict["all_answer_input"] = all_prompts
            
    return dataset_dict

@log_method
def answer_inference(model, tokenizer, answer_data):

     
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=50, 
        do_sample=True)

    texts = []
    index_dict = []
    for key, value in answer_data.items():
        if '.json' in key:
            texts += value['Answer Prediction Prompt Dataset']
            for each_instance_idx in range(len(value['Instances'])):
                index_dict.append((key, each_instance_idx))

            assert len(value['Answer Prediction Prompt Dataset']) == len(value['Instances'])

    result = []


    for each in tqdm.tqdm(texts,miniters=50):
        res = pipeline(each)
        output_text = res[0]['generated_text'][len(each):]
        truncated_result = output_text.strip()
        # result.append(truncated_result)
        # print(result)
        result.append(truncated_result)

    for i, text in enumerate(texts):
        task, index = index_dict[i]
        # Write answer prediction to json file.
        answer_data[task]["Instances"][index]['answer_prediction'] = result[i]

    answer_data["answer_labels"] = result

    del pipeline

    return answer_data