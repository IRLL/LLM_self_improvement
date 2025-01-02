"""
Author: Qianxi Li
Date: June 2, 2024
Description:
    This script orchestrates a multi-iteration training and evaluation process 
    that leverages various prompting strategies, feedback generation, and 
    evaluations on multiple datasets (Natural Instruction, BoolQ, SQuAD, GSM8K). 
    The script handles:
        - Parsing arguments
        - Creating folders for each iteration
        - Constructing prompts for answer generation and feedback
        - Running inference pipelines
        - Optionally performing clustering for prompt optimization
        - Fine-tuning and evaluation on various datasets
"""

import json
import logging
import os
import time
import sys
import torch

from datetime import datetime
from utils import parse_arguments, read_json, write_json  # Custom utilities for argument parsing and JSON IO.
from prompt_compose_helpers import construct_answer_prompts, construct_feedback_prompts  # Helpers for prompt construction.
from corrupted_prompt_compose_helpers import (
    construct_answer_prompts_corrupted,
    construct_feedback_prompts_corrupted
)  # Helpers for corrupted prompt construction.

import warnings

# Configure the logging system globally.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress specific warnings from PyTorch to keep logs clean.
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

# Set debugging mode to False. This might enable certain debug behaviors if set to True.
debug = False

# Enable cuDNN and benchmark for potential performance improvements in PyTorch.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Build a string with comma-separated GPU indices if multiple GPUs are available.
device_str = ','.join([str(i) for i in range(torch.cuda.device_count())])


def main():
    """
    main():
        The entry point function that runs the entire prompt-based 
        iteration workflow, including:
            1. Argument parsing
            2. Directory creation for experiments
            3. Prompt construction for answers and feedback
            4. Inference and fine-tuning pipelines
            5. Various evaluation routines
    """
    # Load arguments.
    args = parse_arguments()  # Parse command-line or config arguments.

    # Store the base path where experimental results will be stored.
    experiment_root_path = args.experiment_root_path

    # Check if the experiment root path exists; if not, create it.
    if not os.path.exists(experiment_root_path):
        os.makedirs(experiment_root_path)

    # Convert the current iteration argument to integer.
    iteration_version = int(args.cur_iteration)

    # Initialize a dictionary to store example prompts.
    prompt_example_dict = {}

    # Create a folder for the current iteration based on 'iteration_version'.
    cur_iter_root_path = os.path.join(experiment_root_path, str(iteration_version))
    os.makedirs(cur_iter_root_path)

    # Create a dedicated folder to store adapter models (fine-tuned models).
    model_adapter_save_path = os.path.join(experiment_root_path, 'adapters')
    if not os.path.exists(model_adapter_save_path):
        os.makedirs(model_adapter_save_path)

    # Build a path to save the adapter for the next model iteration.
    full_model_adapter_save_path = os.path.join(model_adapter_save_path, f"model{iteration_version+1}")
    os.makedirs(full_model_adapter_save_path)

    # Define file paths for storing various outputs in this iteration.
    feedback_dataset_path = os.path.join(cur_iter_root_path, "feedback_dataset.json")
    feedback_prompt_dataset_path = os.path.join(cur_iter_root_path, "feedback_prompts.json")
    prompt_example_dict_path = os.path.join(cur_iter_root_path, "prompt_examples.json")
    new_example_indices_dict_path = os.path.join(cur_iter_root_path, "new_example_indices_dict.json")
    math_result_path = os.path.join(cur_iter_root_path, "math.json")
    fb_major_voting_path = os.path.join(cur_iter_root_path, "major_voting.json")
    squad_response_gen_file = os.path.join(cur_iter_root_path, "squad_reponse_prediction.json")
    squad_eval_result_path = os.path.join(cur_iter_root_path, "squad_eval_result.json")
    adapters_path = os.path.join(experiment_root_path, "adapters")

    # Check iteration_version to decide prompt creation strategy.
    if iteration_version == 0:
        # For the very first iteration, optionally use human-provided examples.
        pos_example = None
        neg_example = None
        if args.enable_initial_human_examples:
            pos_example = args.pos_example_amount  # Number of positive examples from humans.
            neg_example = args.neg_example_amount  # Number of negative examples from humans.

        # If mismatch examples are enabled, use the corrupted prompt constructor.
        if args.enable_mismatch_initial_human_examples:
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts_corrupted(
                args.base_dataset_path,
                args.per_task_data_rows,
                "human",
                prompt_example_dict,
                pos_example,
                neg_example
            )
        else:
            # Otherwise, use the standard prompt construction.
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(
                args.base_dataset_path,
                args.per_task_data_rows,
                "human",
                prompt_example_dict,
                pos_example,
                neg_example
            )

        # If initial human examples are enabled, save them for reference.
        if args.enable_initial_human_examples:
            with open(os.path.join(cur_iter_root_path, "initial_prompt_examples.json"), 'w') as obj:
                obj.write(json.dumps(prompt_example_dict))
    else:
        # For subsequent iterations, if prompt optimization is enabled:
        if args.enable_prompt_optimization:
            # Load previous iteration's prompt examples.
            previous_example_dict_path = os.path.join(
                experiment_root_path, str(iteration_version - 1), "prompt_examples.json"
            )
            prompt_example_dict = read_json(previous_example_dict_path)

            # Construct answer prompts using the loaded examples (LLM-based).
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(
                args.base_dataset_path,
                args.per_task_data_rows,
                "llm",
                prompt_example_dict
            )
        else:
            # Otherwise, revert to human-based prompts if specified.
            answer_prompt_dataset, prompt_example_dict = construct_answer_prompts(
                args.base_dataset_path,
                args.per_task_data_rows,
                "human",
                prompt_example_dict,
                args.pos_example_amount,
                args.neg_example_amount
            )

    # Paths to store the final answer prompts and the dataset generated from them.
    answer_prompt_dataset_path = os.path.join(cur_iter_root_path, "answer_prompts.json")
    answer_dataset_path = os.path.join(cur_iter_root_path, "answer_dataset.json")

    # Write the constructed answer prompt dataset to disk.
    write_json(answer_prompt_dataset_path, answer_prompt_dataset)

    # Prepare arguments for the answer inference script.
    answer_inference_args_json = {
        "cur_iteration": args.cur_iteration,
        "debug": debug,
        "adapters_path": adapters_path,
        "model_path": args.model_path,
        "inference_batch_size": args.eval_inference_batch_size,
        "answer_prompts_path": answer_prompt_dataset_path,
        "answer_dataset_path": answer_dataset_path
    }

    # Construct the system command to run the answer inference.
    str1 = f"CUDA_VISIBLE_DEVICES={device_str} python answer_inference.py '{json.dumps(answer_inference_args_json)}'"
    exit_code = os.system(str1)  # Execute the inference script.

    # Check exit code to handle any failures.
    if exit_code != 0:
        logging.error(f"Answer inference failed with exit code {exit_code}")
        sys.exit(1)

    # Load the newly generated answer dataset for feedback prompt generation.
    answer_dataset = read_json(answer_dataset_path)

    # If mismatch examples were used, create corrupted feedback prompts; otherwise standard prompts.
    if args.enable_mismatch_initial_human_examples:
        feedback_prompt_data = construct_feedback_prompts_corrupted(prompt_example_dict, answer_dataset)
    else:
        feedback_prompt_data = construct_feedback_prompts(prompt_example_dict, answer_dataset)

    # Save feedback prompt data to disk.
    with open(feedback_prompt_dataset_path, "w") as obj:
        obj.write(json.dumps(feedback_prompt_data))

    # Prepare arguments for the feedback inference script.
    feedback_inference_args_json = {
        "cur_iteration": args.cur_iteration,
        "debug": debug,
        "num_return_seq": args.num_return_seq,
        "contamination": args.contamination,
        "adapters_path": adapters_path,
        "model_path": args.model_path,
        "inference_batch_size": 1,
        "feedback_prompts_path": feedback_prompt_dataset_path,
        "feedback_dataset_path": feedback_dataset_path,
        "major_voting_save_path": fb_major_voting_path
    }

    # Construct the command to run feedback inference.
    exit_code = os.system(
        f"CUDA_VISIBLE_DEVICES={device_str} python feedback_inference.py '{json.dumps(feedback_inference_args_json)}'"
    )
    # Check exit code to handle any failures.
    if exit_code != 0:
        logging.error(f"Feedback inference failed with exit code {exit_code}")
        sys.exit(1)

    # If prompt optimization is enabled, perform clustering on the feedback results.
    if args.enable_prompt_optimization:
        example_clustering_args_json = {
            "experiment_root_path": cur_iter_root_path,
            "k": args.clusters,
            "prompt_example_dict_path": prompt_example_dict_path,
            "feedback_dataset_path": feedback_dataset_path
        }
        exit_code = os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python example_clustering.py '{json.dumps(example_clustering_args_json)}'"
        )
        if exit_code != 0:
            logging.error(f"Example clustering failed with exit code {exit_code}")
            sys.exit(1)

    # Prepare arguments for fine-tuning script.
    finetune_arguments_json = {
        "cur_iteration": args.cur_iteration,
        "adapters_path": adapters_path,
        "model_path": args.model_path,
        "feedback_dataset_path": feedback_dataset_path,
        "finetune_eval_data_path": args.na_ins_evalset_path,
        "model_adapter_save_path": full_model_adapter_save_path,
        "result_save_path": cur_iter_root_path
    }

    # Run the fine-tuning script.
    exit_code = os.system(
        f"CUDA_VISIBLE_DEVICES={device_str} python finetune.py '{json.dumps(finetune_arguments_json)}'"
    )
    if exit_code != 0:
        logging.error(f"Finetune failed with exit code {exit_code}")
        sys.exit(1)

    # If Natural Instructions evaluation is enabled, run the evaluation script.
    if args.enable_natural_ins:
        natural_args_json = {
            "cur_iteration": args.cur_iteration,
            "adapters_path": adapters_path,
            "model_path": args.model_path,
            "natural_ins_eval_result_path": os.path.join(cur_iter_root_path, "natural_eval_result.json"),
            "inference_batch_size": args.eval_inference_batch_size,
            "natural_ins_eval_path": args.na_ins_evalset_path
        }
        exit_code = os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python eval_natural_ins.py '{json.dumps(natural_args_json)}'"
        )
        if exit_code != 0:
            logging.error(f"Natural instruction evaluation failed with exit code {exit_code}")
            sys.exit(1)

    # If BoolQ evaluation is enabled, run the evaluation script.
    if args.enable_boolq_eval:
        boolq_args_json = {
            "cur_iteration": args.cur_iteration,
            "adapters_path": adapters_path,
            "model_path": args.model_path,
            "boolq_eval_result_path": os.path.join(cur_iter_root_path, "boolq_eval_result.json"),
            "inference_batch_size": args.eval_inference_batch_size,
            "boolq_eval_path": args.boolq_eval_path
        }
        exit_code = os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python eval_boolq.py '{json.dumps(boolq_args_json)}'"
        )
        if exit_code != 0:
            logging.error(f"BoolQ evaluation failed with exit code {exit_code}")
            sys.exit(1)

    # If SQuAD evaluation is enabled, run the evaluation script.
    if args.enable_squad_eval:
        squad_args_json = {
            "cur_iteration": args.cur_iteration,
            "adapters_path": adapters_path,
            "model_path": args.model_path,
            "squad_eval_result_path": squad_eval_result_path,
            "squad_response_gen_file": squad_response_gen_file,
            "inference_batch_size": args.eval_inference_batch_size,
            "transformed_squad_eval_set_path": args.transformed_squad_eval_set_path,
            "original_squad_eval_set_path": args.original_squad_eval_set_path
        }
        exit_code = os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python eval_squad.py '{json.dumps(squad_args_json)}'"
        )
        if exit_code != 0:
            logging.error(f"SQuAD evaluation failed with exit code {exit_code}")
            sys.exit(1)

    # If GSM8K (math) evaluation is enabled, run the evaluation script.
    if args.enable_gsm8k_eval:
        math_args_json = {
            "cur_iteration": args.cur_iteration,
            "adapters_path": adapters_path,
            "model_path": args.model_path,
            "inference_batch_size": args.eval_inference_batch_size,
            "gsm8k_eval_result_path": math_result_path,
            "gsm8k_eval_path": args.gsm8k_testset
        }
        exit_code = os.system(
            f"CUDA_VISIBLE_DEVICES={device_str} python eval_math.py '{json.dumps(math_args_json)}'"
        )
        if exit_code != 0:
            logging.error(f"GSM8K evaluation failed with exit code {exit_code}")
            sys.exit(1)


if __name__ == "__main__":
    # Invoke the main function when this script is run directly.
    main()

