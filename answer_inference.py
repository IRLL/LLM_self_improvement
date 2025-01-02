"""
Author: Qianxi Li
Date: June 2, 2024
Description: This module handles the inference process for answer prediction using a transformer model.
"""

import transformers
import torch
import os
import tqdm
import json
import sys
import logging
from torch import LongTensor, FloatTensor
from utils import log_method, ClearCache, load_tokenizer, load_model_with_adapters, read_json, write_json, split_into_batches
from transformers import StoppingCriteria, StoppingCriteriaList

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inference(model, tokenizer, batch_input_text, stopping_criteria):
    """
    Perform inference on a batch of input text using the provided model and tokenizer.
    
    Args:
        model: The transformer model for inference
        tokenizer: The tokenizer for processing input text
        batch_input_text: List of input text to process
        stopping_criteria: Criteria for stopping text generation
        
    Returns:
        list: Generated text responses
    """
    # Tokenize input text and move to GPU
    input_ids = tokenizer(batch_input_text, return_tensors="pt", padding=True, truncation=True).to('cuda:0')
    
    # Generate responses with specified parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'],
            do_sample=True,
            use_cache=True,
            num_return_sequences=1,
            max_new_tokens=100,
            temperature=0.3,
            attention_mask=input_ids['attention_mask'],
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria
        )
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Decode generated outputs
    res = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]
    
    # Clean up memory
    del input_ids
    
    return res

@log_method
def answer_inference():
    """
    Main function to perform answer inference using the trained model.
    Processes command line arguments and generates predictions for given prompts.
    """
    # Parse command line arguments
    arguments = json.loads(sys.argv[1])
    iteration = int(arguments['cur_iteration'])
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    answer_prompts_path = arguments['answer_prompts_path']
    answer_dataset_path = arguments['answer_dataset_path']
    inference_batch_size = int(arguments['inference_batch_size'])

    with ClearCache():
        # Load answer data from file
        answer_data = read_json(answer_prompts_path)
        logger.info(f"Loaded answer data from {answer_prompts_path}")

        # Initialize tokenizer and stopping criteria
        tokenizer = load_tokenizer(model_path)
        stop_list = [" \n\n", "\n\n"]
        stop_token_ids = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
        stop_token_ids = [LongTensor(x).to('cuda:0') for x in stop_token_ids]

        class StopOnTokens(StoppingCriteria):
            """Custom stopping criteria for text generation."""
            def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
                for stop_ids in stop_token_ids:
                    if (input_ids[0][-len(stop_ids[0])+1:] == stop_ids[0][1:]).all():
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        
        # Load model with adapters
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        model.eval()

        # Prepare input texts and tracking indices
        texts = []
        index_dict = []
        for key, value in answer_data.items():
            texts += value['Answer Prediction Prompt Dataset']
            for each_instance_idx in range(len(value['Instances'])):
                index_dict.append((key, each_instance_idx))
            
            # Verify data consistency
            assert len(value['Answer Prediction Prompt Dataset']) == len(value['Instances'])

        # Process batches and generate predictions
        result = []
        batches = split_into_batches(texts, inference_batch_size)
        logger.info(f"Processing {len(batches)} batches")

        for batch_idx, each_batch in enumerate(tqdm.tqdm(batches)):
            # Generate predictions for current batch
            res = inference(model, tokenizer, each_batch, stopping_criteria)
            
            # Extract relevant portion of generated text
            for idx, each_output in enumerate(res):
                output_text = each_output[len(each_batch[idx]):]
                truncated_result = output_text.strip()
                result.append(truncated_result)

        # Update answer data with predictions
        for i, text in enumerate(texts):
            task, index = index_dict[i]
            answer_data[task]["Instances"][index]['answer_prediction'] = result[i]

        # Clean up memory
        del result, texts, index_dict

        # Save updated answer data
        write_json(answer_dataset_path, answer_data)
        logger.info(f"Saved predictions to {answer_dataset_path}")

if __name__ == "__main__":
    answer_inference()