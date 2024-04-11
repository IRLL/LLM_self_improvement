import transformers
import torch
import os,tqdm,json,time

import random,copy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import IsolationForest
from utils import log_method

def major_vote_response(model, tokenizer, responses, contamination, batch_size):
    def find_most_centered_data(center, reduced_vectors):
        distances = np.linalg.norm(reduced_vectors - center, axis=1)
        idx = np.argmin(distances)

        return idx

    def batch_encode_strings(bert, tokenizer, strings, batch_size):
        model = bert
        model.eval()  # Set the model to evaluation mode

        # Initialize list to store the vectors
        vectors = []

        # Process strings in batches
        for i in range(0, len(strings), batch_size):
            # Prepare batch data
            batch = strings[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)

            # Move tensors to the device where the model is located
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Get hidden states
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # Use the last hidden states

            # Compute mean of all token embeddings for each sentence in the batch
            batch_vectors = hidden_states.mean(dim=1)
            vectors.extend(batch_vectors)

        # Convert the list of tensors to a single tensor
        vectors = torch.stack(vectors)
        return vectors

    # Encode responses to get the vectors
    response_vectors = batch_encode_strings(
        model, tokenizer, responses, batch_size)

    # # Convert list of tensors to a single tensor
    # response_vectors = torch.stack(encoded_responses)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    response_vectors_2d = pca.fit_transform(response_vectors.numpy())

    # Save the 2D vectors as a torch tensor locally

    # Outlier detection
    iso_forest = IsolationForest(contamination=contamination)
    outliers = iso_forest.fit_predict(response_vectors_2d)
    valid_indices = [i for i, x in enumerate(outliers) if x == 1]
    invalid_indices = [i for i, x in enumerate(outliers) if x == -1]

    pure_vectors = response_vectors_2d[valid_indices]
    outliers = response_vectors_2d[invalid_indices]
    responses = [responses[i] for i in valid_indices]
    # Clustering
    kmeans = KMeans(n_clusters=1,n_init='auto')
    cluster_labels = kmeans.fit_predict(pure_vectors)

    # Identify cluster center
    cluster_center = kmeans.cluster_centers_[0]

    selected_idx = find_most_centered_data(cluster_center, pure_vectors)
    data_center = pure_vectors[selected_idx]
    selected = responses[selected_idx]

    return selected, outliers, response_vectors_2d, cluster_center, data_center, pure_vectors

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
                prompt += f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["reason"]} So the answer is: {each_example["output"]}\n\n"""
            
        if negative_example_amount:
            neg_example = task_examples["Negative Examples"][:negative_example_amount]
            examples += neg_example

            if '-' not in neg_example:
                for each_example in neg_example:
                    prompt +=  f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["reason"]} So the answer is: {each_example["output"]}\n\n"""


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
def answer_inference(model, tokenizer, answer_data, contamination, num_return_seq,bert_model,
                       bert_tokenizer):

     
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="sequential",
        max_new_tokens=50, 
        do_sample=True,
        num_return_sequences=num_return_seq
        )

    texts = []
    index_dict = []
    for key, value in answer_data.items():
        if '.json' in key:
            texts += value['Answer Prediction Prompt Dataset']
            for each_instance_idx in range(len(value['Instances'])):
                index_dict.append((key, each_instance_idx))

            assert len(value['Answer Prediction Prompt Dataset']) == len(value['Instances'])

    result = []
    log_counter=0
    major_voting_log = []


    for each in tqdm.tqdm(texts):
        res = pipeline(each)
        truncated_result = [res[i]['generated_text'][len(each):].split(
                        '\n\n')[0].strip() for i in range(len(res))]
        # result.append(truncated_result)
        # print(result)
        if len(truncated_result) == 1:
            result.append(truncated_result[0])
            voted_feedback= truncated_result
            
        else: 
            voted_feedback, outliers, response_vectors_2d, cluster_center, data_center,pure_vector2d = major_vote_response(
                bert_model, bert_tokenizer, truncated_result, contamination=contamination, batch_size=num_return_seq)
            result.append(voted_feedback)
            if log_counter<20:
                tmp = {"each_feedback_prompt":each,
                        "truncated_result":truncated_result,
                        "outliers":outliers.tolist(),
                        "response_vectors_2d":response_vectors_2d.tolist(), 
                        "cluster_center":cluster_center.tolist(),
                        "data_center":data_center.tolist(),
                        "pure_vector2d":pure_vector2d.tolist()}
                major_voting_log.append(tmp)
                log_counter+=1

    for i, text in enumerate(texts):
        task, index = index_dict[i]
        # Write answer prediction to json file.
        answer_data[task]["Instances"][index]['answer_prediction'] = result[i]

    answer_data["answer_labels"] = result

    del pipeline

    return answer_data, major_voting_log