"""
Author: Qianxi Li
Date: June 4, 2024
Description: This module provides helper functions for composing corrupted prompts for answer and feedback generation.
"""

import json
import os
import copy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compose_examples(example_source, task_examples, pos_example_amount, negative_example_amount):
    """
    Compose example prompts from task examples.
    
    Args:
        example_source (str): Source of examples ("human" or other)
        task_examples (dict): Dictionary containing task examples
        pos_example_amount (int): Number of positive examples to include
        negative_example_amount (int): Number of negative examples to include
        
    Returns:
        tuple: (str, list) Composed prompt and examples list
    """
    prompt = ""
    examples = []
    
    if example_source == "human":
        # Process positive examples
        if pos_example_amount:
            pos_examples = task_examples["Positive Examples"][:pos_example_amount]
            examples.extend(pos_examples)
            # Deliberately corrupt the examples order
            for idx, example in enumerate(pos_examples):
                prompt += f"""### Task:\n{pos_examples[len(pos_examples)-1-idx]["input"]}\n\n### Answer:\n{example["output"]}\n\n"""

        # Process negative examples
        if negative_example_amount:
            neg_examples = task_examples["Negative Examples"][:negative_example_amount]
            examples.extend(neg_examples)
            if '-' not in neg_examples:
                for example in neg_examples:
                    prompt += f"""### Task:\n{example["input"]}\n\n### Answer:\n{example["output"]}\n\n"""
    else:
        # Process regular examples
        examples = task_examples
        for example in task_examples:
            prompt += f"""### Task:\n{example["input"]}\n\n### Answer:\n{example["output"]}\n\n"""

    return prompt, examples

def construct_answer_prompts_corrupted(base_dataset_path, per_task_data_row_amount, 
                                     example_source, prompt_example_dict,
                                     pos_example_amount=None, neg_example_amount=None):
    """
    Construct corrupted answer prompts from dataset.
    
    Args:
        base_dataset_path (str): Path to base dataset
        per_task_data_row_amount (int): Number of data rows per task
        example_source (str): Source of examples
        prompt_example_dict (dict): Dictionary of prompt examples
        pos_example_amount (int, optional): Number of positive examples
        neg_example_amount (int, optional): Number of negative examples
        
    Returns:
        tuple: (dict, dict) Dataset dictionary and current examples dictionary
    """
    dataset_dict = {}
    current_examples_dict = {}
    
    # Process each JSON file in dataset
    for filename in os.listdir(base_dataset_path):
        if ".json" not in filename:
            continue
            
        filepath = os.path.join(base_dataset_path, filename)
        logger.info(f"Processing {filename}")
        
        # Load and process file content
        with open(filepath) as f:
            file_content = json.loads(f.read())
        
        # Create deep copy for modifications
        per_task_dict = copy.deepcopy(file_content)
        
        # Extract task components
        task_definition = file_content["Definition"]
        caution = file_content["Emphasis & Caution"]
        
        # Limit instances according to specified amount
        instance_limit = min(per_task_data_row_amount, len(file_content["Instances"]))
        instances = file_content["Instances"][:instance_limit]
        per_task_dict["Instances"] = instances
        
        # Process examples if specified
        if pos_example_amount or neg_example_amount:
            example_source_data = (file_content["Examples"] if example_source == "human" 
                                 else prompt_example_dict[filename])
            example_prompt, per_task_examples = compose_examples(
                example_source, example_source_data,
                pos_example_amount, neg_example_amount
            )
        
        # Compose prompts for each instance
        per_task_prompt_list = []
        instruction = f"{task_definition} {caution}\n\n"
        context = ("Please refer to the instruction and task information and give your answers. "
                  "You need to follow the examples we provided.")
        
        for instance in instances:
            # Build full prompt
            example_str = (f"### Examples:\n{example_prompt}" 
                         if (pos_example_amount or neg_example_amount) else "")
            full_prompt = (f"### Context:\n{context}\n\n### Instruction:\n{instruction}"
                         f"{example_str}### Task:\n{instance['input']}\n\n### Answer:\n")
            per_task_prompt_list.append(full_prompt)
        
        # Update dictionaries
        per_task_dict["Answer Prediction Prompt Dataset"] = per_task_prompt_list
        if pos_example_amount or neg_example_amount:
            per_task_dict["Current Examples"] = per_task_examples
            current_examples_dict[filename] = per_task_examples
        
        dataset_dict[filename] = per_task_dict
        del per_task_dict
    
    return dataset_dict, current_examples_dict

def compose_feedback_examples(examples):
    """
    Compose feedback examples with deliberate corruption.
    
    Args:
        examples (list): List of example dictionaries
        
    Returns:
        str: Composed feedback examples
    """
    prompt = ""
    for idx, example in enumerate(examples):
        corrupt_one = examples[len(examples)-1-idx]
        prompt += (f"### Task:\n{corrupt_one['input']}\n\n### Predicted Answer:\n"
                  f"{example['output']}\n\n###Feedback:\n{example['reason']}. "
                  f"So the answer should be {example['output']}\n\n")
    return prompt

def construct_feedback_prompts_corrupted(loaded_examples, answer_pred_dataset):
    """
    Construct corrupted feedback prompts from answer predictions.
    
    Args:
        loaded_examples (dict): Dictionary of loaded examples
        answer_pred_dataset (dict): Dataset with answer predictions
        
    Returns:
        dict: Dictionary containing feedback prompts and data
    """
    dataset_dict = {}
    all_task_feedback_input_data = []
    all_task_feedback_gen_prompt_data = []
    
    # Process each task in dataset
    for task_name, task_dict in answer_pred_dataset.items():
        logger.info(f"Processing feedback prompts for task: {task_name}")
        
        # Initialize task-specific lists
        per_task_prompt_list = []
        per_task_no_example_input_list = []
        per_task_dict = copy.deepcopy(task_dict)
        
        # Extract task components
        task_definition = per_task_dict["Definition"]
        caution = per_task_dict["Emphasis & Caution"]
        instances = per_task_dict["Instances"]
        
        # Compose examples if available
        example_prompt = ""
        if loaded_examples:
            example_prompt = compose_feedback_examples(loaded_examples[task_name])
        
        # Prepare context strings
        feedback_prompt = ("Please refer to the instruction and task information, provide your feedback "
                         "for whether you think the predict answer is proper and the reasons. "
                         "You need to follow the examples we provided.")
        context = f"{feedback_prompt}\n\n### Instruction:\n{task_definition} {caution} \n\n"
        
        no_example_feedback = ("Please refer to the instruction and task information, provide your feedback "
                             "for whether you think the predict answer is proper and the reasons.")
        no_example_context = f"{no_example_feedback}\n\n### Instruction:\n{task_definition} {caution} \n\n"
        
        # Process each instance
        for instance in instances:
            # Handle standard answer format
            standard_answer = instance['output']
            if isinstance(standard_answer, list):
                standard_answer = standard_answer[0]
            
            # Compose prompts
            example_str = f"### Examples:\n{example_prompt}" if loaded_examples else ""
            full_prompt = (f"### Context:\n{context}{example_str}### Task:\n{instance['input']}\n\n"
                         f"### Predicted Answer:\n{instance['answer_prediction']}\n\n### Feedback:\n")
            no_example_full_prompt = (f"### Context:\n{no_example_context}### Task:\n{instance['input']}\n\n"
                                    f"### Predicted Answer:\n{instance['answer_prediction']}\n\n### Feedback:\n")
            
            # Add prompts to appropriate lists
            if not loaded_examples:
                per_task_prompt_list.append(no_example_full_prompt)
            else:
                per_task_prompt_list.append(full_prompt)
            per_task_no_example_input_list.append(no_example_full_prompt)
        
        # Update dictionaries
        per_task_dict["Feedback Prediction Prompt Dataset"] = per_task_prompt_list
        dataset_dict[task_name] = per_task_dict
        
        # Extend global lists
        all_task_feedback_input_data.extend(per_task_no_example_input_list)
        all_task_feedback_gen_prompt_data.extend(per_task_prompt_list)
    
    # Add feedback input list to dataset dictionary
    dataset_dict["all_feedback_input_list"] = all_task_feedback_input_data
    
    return dataset_dict