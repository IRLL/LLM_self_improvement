"""
Author: Qianxi Li
Date: June 20, 2024
Description: This script evaluates a model on the SQuAD dataset using a transformer-based model with adapters.
It generates predictions, evaluates their accuracy, and saves the results.
"""

import transformers
import torch
import tqdm
import os
import json
import sys
import logging
from utils import (
    log_method, ClearCache, load_tokenizer, load_model_with_adapters, split_into_batches
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
            max_new_tokens=20,
            attention_mask=input_ids['attention_mask'],
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode generated outputs
    results = [tokenizer.decode(each, skip_special_tokens=True) for each in outputs]

    # Clean up memory
    del input_ids
    return results

@log_method
def eval_squad():
    """
    Evaluate the model on the SQuAD dataset.

    Parses the arguments from the command line, loads the dataset, performs inference,
    evaluates predictions, and saves the results to a file.
    """
    # Parse command-line arguments
    arguments = json.loads(sys.argv[1])
    iteration = int(arguments['cur_iteration'])
    transformed_squad_eval_set_path = arguments['transformed_squad_eval_set_path']
    original_squad_eval_set_path = arguments['original_squad_eval_set_path']
    squad_response_gen_file = arguments['squad_response_gen_file']
    squad_eval_result_path = arguments['squad_eval_result_path']
    adapters_path = arguments['adapters_path']
    model_path = arguments['model_path']   
    inference_batch_size = int(arguments['inference_batch_size'])

    # Clear GPU memory before starting the evaluation
    with ClearCache():
        # Load model and tokenizer
        model = load_model_with_adapters(iteration, adapters_path, model_path)
        tokenizer = load_tokenizer(model_path)
        model.eval()

        # Load evaluation dataset
        with open(transformed_squad_eval_set_path, 'r') as obj:
            squad_data = json.load(obj)

        prompt = """Write a response that appropriately completes answer the question, follow the examples. You should answer 'no answer found' if you cannot find the answer from the context.

        ### Example 1:
        Context: 
        A problem is regarded as inherently difficult if its solution requires significant resources, whatever the algorithm used. The theory formalizes this intuition, by introducing mathematical models of computation to study these problems and quantifying the amount of resources needed to solve them, such as time and storage.

        Question:
        What method is used to intuitively assess or quantify the amount of resources required to solve a computational problem?

        Answer:
        mathematical models of computation

        ### Example 2:
        Context: 
        Under the terms of the Scotland Act 1978, an elected assembly would be set up in Edinburgh provided that the majority of the Scottish electorate voted for it in a referendum to be held on 1 March 1979 that represented at least 40% of the total electorate. The 1979 Scottish devolution referendum to establish a devolved Scottish Assembly failed.

        Question:
        President Wilson committed his government to what in 1974?

        Answer:
        no answer found

        ### Task:
        Context:
        {context}

        Question:
        {question}

        Answer:"""

        # Prepare for batch inference
        batches = split_into_batches(squad_data, inference_batch_size)
        results = {}

        for each_batch in tqdm.tqdm(batches, desc="squad_eval"):
            # Generate prompts for each question
            full_prompt_list = [
                prompt.format(question=item['question'], context=item['context']) 
                for item in each_batch
            ]

            # Perform inference
            responses = inference(model, tokenizer, full_prompt_list)

            for idx, response in enumerate(responses):
                output_text = response[len(full_prompt_list[idx]):].strip()
                answer = output_text

                # Process answers
                if "no answer" in answer.lower() or len(answer) <= 3:
                    answer = ""

                results[each_batch[idx]['id']] = answer

        # Save generated responses to a file
        with open(squad_response_gen_file, 'w') as obj:
            json.dump(results, obj)

        # Evaluate predictions using an external script
        os.system(f"python scripts/official_squad_eval.py --data_file={original_squad_eval_set_path} --pred_file={squad_response_gen_file} --out-file={squad_eval_result_path}")

        # Log final evaluation results
        with open(squad_eval_result_path, 'r') as obj:
            squad_result = json.load(obj)
        logging.info("SQuAD results for iteration %d: %s", iteration, squad_result)

# Execute the evaluation
if __name__ == "__main__":
    eval_squad()
