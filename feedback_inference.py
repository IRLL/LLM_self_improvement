
import transformers
import torch
import os
import tqdm
import json,time
import sys
from utils import log_method,ClearCache,load_tokenizer,split_into_batches,load_model_with_adapters, read_json, write_json,load_bert

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import IsolationForest

from sklearn.impute import SimpleImputer
from transformers import StoppingCriteria,StoppingCriteriaList
from torch import LongTensor, FloatTensor





def major_vote_response(model, tokenizer, responses, contamination, batch_size):
    def find_most_centered_data(center, reduced_vectors):
        distances = np.linalg.norm(reduced_vectors - center, axis=1)
        idx = np.argmin(distances)

        return idx

    def batch_encode_strings(model, tokenizer, strings, batch_size):
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
            input_ids = inputs['input_ids'].to("cuda:0")
            attention_mask = inputs['attention_mask'].to("cuda:0")

            # Get hidden states
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # Use the last hidden states

            # Compute mean of all token embeddings for each sentence in the batch
            batch_vectors = hidden_states.mean(dim=1)
            vectors.extend(batch_vectors)

        # Convert the list of tensors to a single tensor
        vectors = torch.stack(vectors).cpu()
        return vectors

    # Encode responses to get the vectors
    response_vectors = batch_encode_strings(
        model, tokenizer, responses, batch_size)

    imputer = SimpleImputer(strategy='mean')
    response_vectors = imputer.fit_transform(response_vectors)
    # # Convert list of tensors to a single tensor
    # response_vectors = torch.stack(encoded_responses)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    response_vectors_2d = pca.fit_transform(response_vectors)

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


def inference(model, tokenizer, batch_input_text,num_return_sequences,stopping_criteria):
    input_ids = tokenizer(batch_input_text, return_tensors="pt", padding=True, truncation=True).to('cuda:0')
    with torch.no_grad():
        generated_texts = [[] for _ in range(len(batch_input_text))]
        for gen_iter in range(num_return_sequences):
            
            outputs = model.generate(
                input_ids=input_ids['input_ids'], 
                do_sample=True, 
                use_cache=True, 
                num_return_sequences=1,
                max_new_tokens=200,
                #repetition_penalty=1.15,
                attention_mask=input_ids['attention_mask'] ,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria
            )
            #torch.cuda.empty_cache()
            for idx,each_prompt_sampled_response in enumerate(outputs): 
                decoded = tokenizer.decode(each_prompt_sampled_response, skip_special_tokens=True)
                generated_texts[idx].append(decoded)

    torch.cuda.empty_cache()
    del input_ids, outputs
    return generated_texts

@log_method
def feedback_inference():

    arguments = json.loads(sys.argv[1])

    iteration = int(arguments['cur_iteration'])
    num_return_seq = int(arguments['num_return_seq'])
    contamination = float(arguments['contamination'])
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    feedback_prompts_path = arguments['feedback_prompts_path']
    feedback_dataset_path = arguments['feedback_dataset_path']
    major_voting_save_path = arguments['major_voting_save_path']
    inference_batch_size = int(arguments['inference_batch_size'])

    with ClearCache():
        tokenizer = load_tokenizer(model_path)

        stop_list = [" \n\n", "\n\n"]
        stop_token_ids = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
        stop_token_ids = [LongTensor(x).to('cuda:0') for x in stop_token_ids]
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
                for stop_ids in stop_token_ids:
                    if (input_ids[0][-len(stop_ids[0])+1:] == stop_ids[0][1:]).all():
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        feedback_data = read_json(feedback_prompts_path)
        
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        model.eval()
        bert_model, bert_tokenizer = load_bert()

        result = []
        log_counter=0
        major_voting_log = []

        for task_name in tqdm.tqdm(list(feedback_data.keys()), desc="Each task fb generation:", position=0):
            if ".json" in task_name:
                task_dict = feedback_data[task_name]
                per_task_dataset = task_dict["Feedback Prediction Prompt Dataset"]
                per_task_full_string_list = []
                
                batches = split_into_batches(per_task_dataset, inference_batch_size)
                index = 0
                for each_batch in batches:
                    print("start batch generate")
                    time1 = time.time()
                    res = inference(model, tokenizer, each_batch, num_return_seq, stopping_criteria)
                    time2 = time.time()
                    print("diff",time2-time1)
                    assert 1==2
                    for group_idx, each_input_response_group in enumerate(res):

                        input_text = each_batch[group_idx]
                        truncated_result_list = []

                        for each_response in each_input_response_group:
                            truncated_result_list.append(each_response[len(input_text):].split('\n\n')[0].strip())

                        if len(truncated_result_list) == 1:
                            result.append(truncated_result_list[0])
                            per_task_full_string_list.append(input_text+truncated_result_list[0])
                            voted_feedback= truncated_result_list[0]
                            
                        else: 
                            voted_feedback, outliers, response_vectors_2d, cluster_center, data_center,pure_vector2d = major_vote_response(
                                bert_model, bert_tokenizer, truncated_result_list, contamination=contamination, batch_size=num_return_seq)
                            result.append(voted_feedback)
                            per_task_full_string_list.append(input_text+voted_feedback)
                            if log_counter<20:
                                tmp = {"each_feedback_prompt":each_response,
                                        "truncated_result":truncated_result_list,
                                        "outliers":outliers.tolist(),
                                        "response_vectors_2d":response_vectors_2d.tolist(), 
                                        "cluster_center":cluster_center.tolist(),
                                        "data_center":data_center.tolist(),
                                        "pure_vector2d":pure_vector2d.tolist()}
                                major_voting_log.append(tmp)
                                log_counter+=1
                        task_dict['Instances'][index]['fb_pred'] = voted_feedback
                        index+=1
                
                print("batch finish")
                feedback_data[task_name]['Full clustering context'] = per_task_full_string_list
                del batches
                del per_task_full_string_list
                del per_task_dataset

        feedback_data["Feedback Label"] = result

        write_json(feedback_dataset_path, feedback_data)
        write_json(major_voting_save_path, major_voting_log)

feedback_inference()