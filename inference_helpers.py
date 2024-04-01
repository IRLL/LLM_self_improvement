import transformers
import torch
import os,tqdm,json,time

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
        max_new_tokens=50, 
        do_sample=True
        # num_beams=3,
        # num_return_sequences=1,
        #batch_size=1

    )

    texts = []
    index_dict = []
    for key, value in answer_data.items():
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

    del pipeline

    return answer_data

@log_method
def feedback_inference(model, tokenizer, feedback_prompt_data, new_example_indices_dict):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=100,
        do_sample=True
        # num_beams=3,
        # num_return_sequences=1

    )
    result = []
    prompt_example_dict = {}

    feedback_data = feedback_prompt_data
    for task_name in tqdm.tqdm(list(feedback_data.keys()),desc="Each task fb generation:", position=0):
        if ".json" in task_name:
            task_dict = feedback_data[task_name]
            selected_example_index_list = new_example_indices_dict[task_name]
            index = 0
            prompt_example_list = []

            for each_feedback_prompt in tqdm.tqdm(task_dict["Feedback Prediction Prompt Dataset"],desc="Each row of task", position=1, leave=False):
                #reason = f"fake feedback"#pipeline(each)
                res = pipeline(each_feedback_prompt)
                output_text = res[0]['generated_text'][len(each_feedback_prompt):]
                truncated_result = output_text.strip()
                #result.append(truncated_result)

                if index in selected_example_index_list:
                    prompt_example_list.append({"input":task_dict['Instances'][index]["input"],
                                                "output":task_dict['Instances'][index]["answer_prediction"],
                                                "reason":truncated_result})
                result.append(truncated_result)

                index+=1

            prompt_example_dict[task_name] = prompt_example_list

    feedback_data["Feedback Label"] = result

    del pipeline
    return feedback_data, prompt_example_dict