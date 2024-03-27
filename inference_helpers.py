import transformers
import torch
import os,tqdm,json

import random

from utils import log_method

@log_method
def answer_inference(model, tokenizer, answer_data):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=200, 
        num_beams=5,
        num_return_sequences=1

    )

    texts = []
    index_dict = []
    for key, value in answer_data.items():
        texts += value['Answer Prediction Prompt Dataset']
        for each_instance_idx in range(len(value['Instances'])):
            index_dict.append((key, each_instance_idx))

        assert len(value['Answer Prediction Prompt Dataset']) == len(value['Instances'])
    print("texts",len(texts))

    result = []
    idx = 0
    for each in tqdm.tqdm(texts):
        res = f"ahhahahha{random.randint(3,300)}"#pipeline(each)
        # output_text = res['generated_text'][len(texts[idx]):]
        # truncated_result = output_text.strip()
        # result.append(truncated_result)
        # print(result)
        result.append(res)

    for i, text in enumerate(texts):
        task, index = index_dict[i]
        # Write answer prediction to json file.
        answer_data[task]["Instances"][index]['answer_prediction'] = result[i]

    del pipeline

    return answer_data

@log_method
def feedback_inference(model, tokenizer, feedback_prompt_data):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=200,
        num_beams=5,
        num_return_sequences=1

    )

    feedback_data = feedback_prompt_data

    texts = feedback_data["all_task_feedback_gen_prompt_data"]

    result = []
    idx = 0
    for each in tqdm.tqdm(texts):
        res = f"fake feedback"#pipeline(each)
        # output_text = res['generated_text'][len(texts[idx]):]
        # truncated_result = output_text.strip()
        # result.append(truncated_result)
        result.append(res)
        #print(result)

    feedback_data["Feedback Label"]=[]

    for i, text in enumerate(texts):
        # Write answer prediction to json file.
        feedback_data["Feedback Label"].append(result[i])
        #print(f"Input: {text}\nOutput: {result[i]['generated_text']}\n")


    del pipeline
    return feedback_data