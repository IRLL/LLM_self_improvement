"""
Author: Qianxi Li
Date: June 20, 2024
Description: This script evaluates a model on the Natural Instructions dataset using a transformer-based model with adapters. It includes a custom inference method for generating predictions.

"""

import transformers
import torch
import os
import tqdm
import json
import sys
import logging
from torchmetrics.text.rouge import ROUGEScore
from utils import (
    log_method, ClearCache, load_model_with_adapters, load_tokenizer, split_into_batches
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inference(model, tokenizer, batch_input_text):
    """
    Perform inference on a batch of input text.

    Args:
        model: Transformer model for text generation.
        tokenizer: Tokenizer associated with the model.
        batch_input_text: List of input texts for processing.

    Returns:
        List of generated responses for the input text batch.
    """
    # Tokenize input texts
    input_ids = tokenizer(
        batch_input_text, 
        return_tensors="pt",
        max_length=2048,
        padding=True,
        truncation=True
    ).to('cuda:0')

    # Perform model inference without gradients
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'],
            do_sample=True,
            use_cache=True,
            num_return_sequences=1,
            max_new_tokens=100,
            attention_mask=input_ids['attention_mask'],
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode generated outputs
    results = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]

    # Clean up memory
    del input_ids
    return results

@log_method
def eval_natural_ins():
    """
    Evaluate the model on the Natural Instructions dataset.

    Parses the arguments from the command line, loads the dataset, performs inference,
    calculates ROUGE metrics, and saves the results to a file.
    """
    # Parse command-line arguments
    arguments = json.loads(sys.argv[1])
    iteration = int(arguments['cur_iteration'])
    natural_ins_eval_result_path = arguments['natural_ins_eval_result_path']
    natural_ins_eval_path = arguments['natural_ins_eval_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    inference_batch_size = int(arguments['inference_batch_size'])

    # Initialize ROUGE score calculator
    rouge = ROUGEScore()

    # Clear GPU memory before starting the evaluation
    with ClearCache():
        # Load model and tokenizer
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        # Load evaluation dataset
        with open(natural_ins_eval_path, 'r') as obj:
            natural_ins_data = json.load(obj)

        # Extract labels from dataset
        labels = [item['label'] for item in natural_ins_data]

        # Prepare for batch inference
        batches = split_into_batches(natural_ins_data, inference_batch_size)
        predictions = []

        for each_batch in tqdm.tqdm(batches, desc="natural_ins_eval"):
            # Generate prompts for each input
            full_prompt_list = [item['input'] for item in each_batch]

            # Perform inference
            results = inference(model, tokenizer, full_prompt_list)
            for idx, each_output in enumerate(results):
                # Extract the model's output
                output_text = each_output[len(full_prompt_list[idx]):].strip()
                predictions.append(output_text)

        # Calculate ROUGE metrics
        metrics = rouge(predictions, labels)
        metrics = {k: v.item() for k, v in metrics.items()}
        logging.info("Natural Instructions metrics: %s", metrics)

        # Save the metrics to a file
        with open(natural_ins_eval_result_path, 'w') as obj:
            json.dump(metrics, obj)

        # Clean up memory
        del labels, predictions, natural_ins_data

# Execute the evaluation
if __name__ == "__main__":
    eval_natural_ins()
