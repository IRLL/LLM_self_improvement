"""
Author: Qianxi Li
Date: June 13, 2024
Description: 
Feedback Inference Module

This module implements feedback inference using transformer models with adapter support.
It includes functionality for response generation, major voting, and outlier detection.
"""

import logging
import json
import sys
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from torch import LongTensor, FloatTensor
from transformers import StoppingCriteria, StoppingCriteriaList

from utils import (
    log_method,
    ClearCache,
    load_tokenizer,
    split_into_batches,
    load_model_with_adapters,
    read_json,
    write_json,
    load_bert
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def major_vote_response(
    model: Any,
    tokenizer: Any,
    responses: List[str],
    contamination: float,
    batch_size: int
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform major voting on model responses using clustering and outlier detection.

    Args:
        model: The BERT model for encoding responses
        tokenizer: The BERT tokenizer
        responses: List of response strings to analyze
        contamination: Contamination parameter for outlier detection
        batch_size: Batch size for processing

    Returns:
        Tuple containing:
        - Selected response string
        - Outlier vectors
        - 2D response vectors
        - Cluster center
        - Data center
        - Pure vectors
    """
    def find_most_centered_data(center: np.ndarray, reduced_vectors: np.ndarray) -> int:
        """Find the index of the vector closest to the center."""
        # Calculate Euclidean distances from each vector to the center
        distances = np.linalg.norm(reduced_vectors - center, axis=1)
        # Return index of minimum distance
        return np.argmin(distances)

    def batch_encode_strings(
        model: Any,
        tokenizer: Any,
        strings: List[str],
        batch_size: int
    ) -> torch.Tensor:
        """Encode strings in batches using the model."""
        # Set model to evaluation mode
        model.eval()
        vectors = []

        # Process strings in batches
        for i in range(0, len(strings), batch_size):
            batch = strings[i:i + batch_size]
            # Tokenize input batch
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            # Move tensors to GPU
            input_ids = inputs['input_ids'].to("cuda:0")
            attention_mask = inputs['attention_mask'].to("cuda:0")

            # Generate embeddings
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state

            # Average token embeddings for each sentence
            batch_vectors = hidden_states.mean(dim=1)
            vectors.extend(batch_vectors)

        # Combine all vectors
        return torch.stack(vectors).cpu()

    # Encode all responses into vectors
    response_vectors = batch_encode_strings(model, tokenizer, responses, batch_size)

    # Handle missing values in vectors
    imputer = SimpleImputer(strategy='mean')
    response_vectors = imputer.fit_transform(response_vectors)

    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    response_vectors_2d = pca.fit_transform(response_vectors)

    # Detect outliers
    iso_forest = IsolationForest(contamination=contamination)
    outliers = iso_forest.fit_predict(response_vectors_2d)
    valid_indices = [i for i, x in enumerate(outliers) if x == 1]
    invalid_indices = [i for i, x in enumerate(outliers) if x == -1]

    # Separate valid and invalid vectors
    pure_vectors = response_vectors_2d[valid_indices]
    outliers = response_vectors_2d[invalid_indices]
    responses = [responses[i] for i in valid_indices]

    # Cluster valid responses
    kmeans = KMeans(n_clusters=1, n_init='auto')
    kmeans.fit_predict(pure_vectors)

    # Find cluster center and most central response
    cluster_center = kmeans.cluster_centers_[0]
    selected_idx = find_most_centered_data(cluster_center, pure_vectors)
    data_center = pure_vectors[selected_idx]
    selected = responses[selected_idx]

    return selected, outliers, response_vectors_2d, cluster_center, data_center, pure_vectors

def inference(
    model: Any,
    tokenizer: Any,
    batch_input_text: List[str],
    num_return_sequences: int,
    stopping_criteria: StoppingCriteriaList
) -> List[List[str]]:
    """
    Generate responses for a batch of input texts.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        batch_input_text: List of input texts
        num_return_sequences: Number of sequences to generate per input
        stopping_criteria: Criteria for stopping generation

    Returns:
        List of lists containing generated texts
    """
    # Tokenize inputs
    input_ids = tokenizer(
        batch_input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to('cuda:0')

    generated_texts = [[] for _ in range(len(batch_input_text))]

    # Generate responses
    with torch.no_grad():
        for _ in range(num_return_sequences):
            outputs = model.generate(
                input_ids=input_ids['input_ids'],
                do_sample=True,
                use_cache=True,
                num_return_sequences=1,
                max_new_tokens=200,
                attention_mask=input_ids['attention_mask'],
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria
            )

            # Decode and store generated texts
            for idx, each_prompt_sampled_response in enumerate(outputs):
                decoded = tokenizer.decode(
                    each_prompt_sampled_response,
                    skip_special_tokens=True
                )
                generated_texts[idx].append(decoded)

    # Clean up GPU memory
    torch.cuda.empty_cache()
    del input_ids, outputs

    return generated_texts

@log_method
def feedback_inference() -> None:
    """
    Main function for generating and processing feedback using the model.
    Loads configuration from command line arguments and processes feedback data.
    """
    # Load configuration from command line
    arguments = json.loads(sys.argv[1])
    
    # Extract configuration parameters
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
        # Initialize models and tokenizers
        tokenizer = load_tokenizer(model_path)
        
        # Set up stopping criteria
        stop_list = [" \n\n", "\n\n"]
        stop_token_ids = [
            tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids']
            for x in stop_list
        ]
        stop_token_ids = [LongTensor(x).to('cuda:0') for x in stop_token_ids]

        class StopOnTokens(StoppingCriteria):
            """Custom stopping criteria for text generation."""
            def __call__(
                self,
                input_ids: LongTensor,
                scores: FloatTensor,
                **kwargs
            ) -> bool:
                """Check if generation should stop based on token sequence."""
                for stop_ids in stop_token_ids:
                    if (input_ids[0][-len(stop_ids[0])+1:] == stop_ids[0][1:]).all():
                        return True
                return False

        # Initialize models and data
        stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        feedback_data = read_json(feedback_prompts_path)
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        model.eval()
        bert_model, bert_tokenizer = load_bert()

        result = []
        log_counter = 0
        major_voting_log = []

        # Process each task
        for task_name in tqdm.tqdm(
            list(feedback_data.keys()),
            desc="Processing feedback generation tasks",
            position=0
        ):
            if ".json" in task_name:
                task_dict = feedback_data[task_name]
                per_task_dataset = task_dict["Feedback Prediction Prompt Dataset"]
                per_task_full_string_list = []
                
                # Process task in batches
                batches = split_into_batches(per_task_dataset, inference_batch_size)
                index = 0
                
                for each_batch in batches:
                    # Generate responses for batch
                    res = inference(
                        model,
                        tokenizer,
                        each_batch,
                        num_return_seq,
                        stopping_criteria
                    )

                    for group_idx, each_input_response_group in enumerate(res):
                        input_text = each_batch[group_idx]
                        truncated_result_list = []

                        # Process each response in the group
                        for each_response in each_input_response_group:
                            truncated_result_list.append(
                                each_response[len(input_text):].split('\n\n')[0].strip()
                            )

                        # Handle single response case
                        if len(truncated_result_list) == 1:
                            voted_feedback = truncated_result_list[0]
                            result.append(voted_feedback)
                            per_task_full_string_list.append(input_text + voted_feedback)
                        else:
                            # Perform major voting for multiple responses
                            voted_feedback, outliers, response_vectors_2d, cluster_center, data_center, pure_vector2d = major_vote_response(
                                bert_model,
                                bert_tokenizer,
                                truncated_result_list,
                                contamination=contamination,
                                batch_size=num_return_seq
                            )
                            
                            result.append(voted_feedback)
                            per_task_full_string_list.append(input_text + voted_feedback)
                            
                            # Log first 20 major voting results
                            if log_counter < 20:
                                tmp = {
                                    "each_feedback_prompt": each_response,
                                    "truncated_result": truncated_result_list,
                                    "outliers": outliers.tolist(),
                                    "response_vectors_2d": response_vectors_2d.tolist(),
                                    "cluster_center": cluster_center.tolist(),
                                    "data_center": data_center.tolist(),
                                    "pure_vector2d": pure_vector2d.tolist()
                                }
                                major_voting_log.append(tmp)
                                log_counter += 1
                                
                        task_dict['Instances'][index]['fb_pred'] = voted_feedback
                        index += 1
                
                logger.info(f"Completed processing batch for task: {task_name}")
                feedback_data[task_name]['Full clustering context'] = per_task_full_string_list
                
                # Clean up memory
                del batches
                del per_task_full_string_list
                del per_task_dataset

        # Store results
        feedback_data["Feedback Label"] = result
        write_json(feedback_dataset_path, feedback_data)
        write_json(major_voting_save_path, major_voting_log)

if __name__ == "__main__":
    feedback_inference()