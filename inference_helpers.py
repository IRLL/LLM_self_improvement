import transformers
import torch
import os
import tqdm
import json
import time

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


@log_method
def answer_inference(model, tokenizer, answer_data, debug):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=50,
        do_sample=True,
        num_return_sequences=1
        # batch_size=1

    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    texts = []
    index_dict = []
    for key, value in answer_data.items():
        texts += value['Answer Prediction Prompt Dataset']
        for each_instance_idx in range(len(value['Instances'])):
            index_dict.append((key, each_instance_idx))

        assert len(value['Answer Prediction Prompt Dataset']
                   ) == len(value['Instances'])

    result = []

    for each in tqdm.tqdm(texts):

        truncated_result = "answer_fake"
        if not debug:
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
def feedback_inference(model,
                       tokenizer,
                       feedback_prompt_data,
                       new_example_indices_dict,
                       num_return_seq,
                       bert_model,
                       bert_tokenizer,
                       contamination,
                       debug):

    n_gpus = torch.cuda.device_count()
    max_memory = {}
    if n_gpus >= 2:

        max_memory[0] = "3GIB"
        max_memory[1] = "16GIB"
    else:
        max_memory[0] = "16GIB"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="sequential",
        # max_memory=max_memory,
        max_new_tokens=100,
        do_sample=True,
        num_return_sequences=num_return_seq

    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

    result = []
    prompt_example_dict = {}
    log_counter=0
    major_voting_log = []


    feedback_data = feedback_prompt_data
    for task_name in tqdm.tqdm(list(feedback_data.keys()), desc="Each task fb generation:", position=0):
        if ".json" in task_name:
            task_dict = feedback_data[task_name]
            selected_example_index_list = new_example_indices_dict[task_name]
            index = 0
            prompt_example_list = []

            for each_feedback_prompt in task_dict["Feedback Prediction Prompt Dataset"]:
                # reason = f"fake feedback"#pipeline(each)
                truncated_result = "feedback_fake"
                if not debug:
                    res = pipeline(each_feedback_prompt)
                    truncated_result = [res[i]['generated_text'][len(each_feedback_prompt):].split(
                        '\n\n')[0].strip() for i in range(len(res))]


                    if len(truncated_result) == 1:
                        result.append(truncated_result[0])
                        voted_feedback= truncated_result
                        
                    else: 
                        voted_feedback, outliers, response_vectors_2d, cluster_center, data_center,pure_vector2d = major_vote_response(
                            bert_model, bert_tokenizer, truncated_result, contamination=contamination, batch_size=num_return_seq)
                        result.append(voted_feedback)
                        if log_counter<20:
                            tmp = {"each_feedback_prompt":each_feedback_prompt,
                                    "truncated_result":truncated_result,
                                    "outliers":outliers.tolist(),
                                    "response_vectors_2d":response_vectors_2d.tolist(), 
                                    "cluster_center":cluster_center.tolist(),
                                    "data_center":data_center.tolist(),
                                    "pure_vector2d":pure_vector2d.tolist()}
                            major_voting_log.append(tmp)
                            log_counter+=1
                # result.append(truncated_result)

                if index in selected_example_index_list:
                    prompt_example_list.append({"input": task_dict['Instances'][index]["input"],
                                                "output": task_dict['Instances'][index]["answer_prediction"],
                                                "reason": voted_feedback})
                

                index += 1

            prompt_example_dict[task_name] = prompt_example_list

    feedback_data["Feedback Label"] = result

    del pipeline
    return feedback_data, prompt_example_dict,major_voting_log
