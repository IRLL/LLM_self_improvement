
import transformers
import torch
import os
import tqdm
import json
import sys
from utils import log_method,ClearCache,load_tokenizer,split_into_batches,load_model_with_adapters, read_json, write_json,load_bert

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import IsolationForest

from sklearn.impute import SimpleImputer

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


def inference(model, tokenizer, batch_input_text,num_return_sequences):
    input_ids = tokenizer(batch_input_text, return_tensors="pt", max_length=2048, padding=True, truncation=True).to('cuda:0')
    with torch.no_grad():
        generated_texts = [[] for _ in range(len(batch_input_text))]
        for gen_iter in range(num_return_sequences):
            
            outputs = model.generate(
                input_ids=input_ids['input_ids'], 
                do_sample=True, 
                use_cache=True, 
                num_return_sequences=1,
                max_new_tokens=100,
                attention_mask=input_ids['attention_mask'] ,
                pad_token_id=tokenizer.pad_token_id
            )
            torch.cuda.empty_cache()
            for idx,each_prompt_sampled_response in enumerate(outputs): 
                decoded = tokenizer.decode(each_prompt_sampled_response, skip_special_tokens=True)
                generated_texts[idx].append(decoded)



            # for i in range(len(batch_input_text)):
            #     # Each input's output starts at index i * num_return_sequences
            #     start_idx = i * num_return_sequences
            #     # Slice out the sequences for the current input
            #     batch_generated_texts = [tokenizer.decode(outputs[j], skip_special_tokens=True) for j in range(start_idx, start_idx + num_return_sequences)]
            #     generated_texts.append(batch_generated_texts)

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
    current_prompt_examples_path = arguments['current_prompt_examples_path']
    major_voting_save_path = arguments['major_voting_save_path']
    new_example_indices_dict_path = arguments['new_example_indices_dict_path']
    inference_batch_size = int(arguments['inference_batch_size'])

    with ClearCache():
        feedback_data = read_json(feedback_prompts_path)
        new_example_indices_dict = read_json(new_example_indices_dict_path)

        tokenizer = load_tokenizer(model_path)
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        model.eval()
        bert_model, bert_tokenizer = load_bert()

        # pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     # torch_dtype=torch.float16,
        #     max_new_tokens=100,
        #     do_sample=True,
        #     num_return_sequences=num_return_seq

        # )
        # pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

        result = []
        prompt_example_dict = {}
        log_counter=0
        major_voting_log = []


        
        for task_name in tqdm.tqdm(list(feedback_data.keys()), desc="Each task fb generation:", position=0):
            if ".json" in task_name:
                task_dict = feedback_data[task_name]
                selected_example_index_list = new_example_indices_dict[task_name]
                index = 0
                prompt_example_list = []
                per_task_dataset = task_dict["Feedback Prediction Prompt Dataset"]
                
                batches = split_into_batches(per_task_dataset, inference_batch_size)

                for each_batch in batches:
                    # reason = f"fake feedback"#pipeline(each)

                    #res = pipeline(each_feedback_prompt)
                    res = inference(model, tokenizer, each_batch,num_return_seq)

                    for group_idx, each_input_response_group in enumerate(res):

                        input_text = each_batch[group_idx]
                        truncated_result_list = []

                        for each_response in each_input_response_group:
                            truncated_result_list.append(each_response[len(input_text):].split('\n\n')[0].strip())

                    # truncated_result = [res[i]['generated_text'][len(each_feedback_prompt):].split(
                    #     '\n\n')[0].strip() for i in range(len(res))]


                        if len(truncated_result_list) == 1:
                            result.append(truncated_result_list[0])
                            voted_feedback= truncated_result_list[0]
                            
                        else: 
                            voted_feedback, outliers, response_vectors_2d, cluster_center, data_center,pure_vector2d = major_vote_response(
                                bert_model, bert_tokenizer, truncated_result_list, contamination=contamination, batch_size=num_return_seq)
                            result.append(voted_feedback)
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

                        if index in selected_example_index_list:
                            prompt_example_list.append({"input": task_dict['Instances'][index]["input"],
                                                        "output": task_dict['Instances'][index]["answer_prediction"],
                                                        "reason": voted_feedback})
                        

                        index += 1

                prompt_example_dict[task_name] = prompt_example_list
                del batches
                del per_task_dataset

        feedback_data["Feedback Label"] = result

        write_json(feedback_dataset_path, feedback_data)
        write_json(current_prompt_examples_path, prompt_example_dict)
        write_json(major_voting_save_path, major_voting_log)

feedback_inference()