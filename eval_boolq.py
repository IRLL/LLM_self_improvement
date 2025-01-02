"""
Author: Qianxi Li
Date: June 20, 2024
Description: This script evaluates a model on the BoolQ dataset using a transformer-based model with adapters. It includes a custom inference method for generating predictions.

"""

import transformers
import torch
import os
import tqdm
import json
import sys
import logging

from utils import (
    calculate_classification_metrics,
    log_method,
    ClearCache,
    load_model_with_adapters,
    load_tokenizer,
    split_into_batches
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inference(model, tokenizer, batch_input_text):
    """
    Perform inference on a batch of input text.

    Args:
        model: The transformer model for text generation.
        tokenizer: Tokenizer associated with the model.
        batch_input_text: List of input texts to process.

    Returns:
        List of generated responses for the input text batch.
    """
    # Tokenize the input text with padding and truncation
    input_ids = tokenizer(
        batch_input_text, 
        return_tensors="pt", 
        max_length=2048, 
        padding=True, 
        truncation=True
    ).to('cuda:0')

    with torch.no_grad():
        # Generate output using the model
        outputs = model.generate(
            input_ids=input_ids['input_ids'], 
            do_sample=True, 
            use_cache=True, 
            num_return_sequences=1,
            max_new_tokens=10,
            attention_mask=input_ids['attention_mask'],
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated outputs
    results = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]

    # Free memory by deleting input IDs
    del input_ids
    return results

@log_method
def eval_boolq():
    """
    Evaluate the model on the BoolQ dataset by generating predictions
    and calculating classification metrics.

    This function processes the dataset in batches, performs inference,
    and saves evaluation metrics to a specified file.
    """
    # Parse arguments passed via the command line
    arguments = json.loads(sys.argv[1])
    iteration = int(arguments['cur_iteration'])
    boolq_eval_result_path = arguments['boolq_eval_result_path']
    boolq_eval_path = arguments['boolq_eval_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']
    inference_batch_size = int(arguments['inference_batch_size'])

    # Clear GPU memory before loading model and tokenizer
    with ClearCache():
        # Load the model and tokenizer
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        # Load the evaluation dataset
        with open(boolq_eval_path) as obj:
            boolq_data = json.loads(obj.read())

        # Define the prompt template
        prompt = """
        Write a response that appropriately answers the question. Your answer should be "True" or "False".

        ### Example 1:
        Passage: 
        The Vampire Diaries, an American supernatural drama, was renewed for an eighth season by The CW on March 11, 2016. On July 23, 2016, the CW announced that the upcoming season would be the series' last and would consist of 16 episodes. The season premiered on October 21, 2016 and concluded on March 10, 2017.

        Question:
        will there be a season 8 of vampire diaries?

        Answer:
        True

        ### Example 2:
        Passage: 
        This is the list of U.S. states that have participated in the Little League World Series. As of the 2018 LLWS, eight states had never reached the LLWS: Alaska, Colorado, Kansas, North Dakota, Utah, Vermont, Wisconsin, and Wyoming; additionally, the District of Columbia has never reached the LLWS.

        Question:
        has wisconsin ever been in the little league world series?

        Answer:
        False

        ### Task:
        Passage:
        {passage}

        Question:
        {question}

        Answer:
        """

        # Prepare for batch inference
        batches = split_into_batches(boolq_data, inference_batch_size)
        predictions = []
        labels = []

        for each_batch in tqdm.tqdm(batches):
            # Generate the full prompt for each data item
            full_prompt_list = [
                prompt.format(question=item['question'], passage=item['passage']) 
                for item in each_batch
            ]

            # Perform inference on the batch
            res = inference(model, tokenizer, full_prompt_list)
            for idx, each_output in enumerate(res):
                # Process the model output to extract predictions
                output_text = each_output[len(full_prompt_list[idx]):]
                truncated_result = output_text.strip()

                # Map results to binary classification
                if "false" in truncated_result.lower():
                    predictions.append(0)
                else:
                    predictions.append(1)
                labels.append(each_batch[idx]["answer"])

        # Calculate evaluation metrics
        metrics = calculate_classification_metrics(predictions, labels)
        logging.info("BoolQ metrics: %s", metrics)

        # Save the metrics to a file
        with open(boolq_eval_result_path, 'w') as obj:
            obj.write(json.dumps(metrics))

        # Free memory by deleting unused variables
        del labels, predictions, boolq_data

# Execute the evaluation function
if __name__ == "__main__":
    eval_boolq()
