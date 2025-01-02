"""
Author: Qianxi Li
Date: June 2, 2024
Description:
    This module provides two main functions for constructing prompts:
        1. construct_answer_prompts: Composes prompts for answer generation 
           based on the given dataset, examples, and iteration parameters.
        2. construct_feedback_prompts: Composes prompts for feedback generation 
           using model predictions and existing examples.

    Both functions return datasets in dictionary form that can be serialized 
    to JSON for further processing.
"""

import json
import os
import copy
import logging

# Configure logging to show INFO level messages and above with a standard format.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def construct_answer_prompts(base_dataset_path,
                             per_task_data_row_amount,
                             example_source,
                             prompt_example_dict,
                             pos_example_amount=None,
                             neg_example_amount=None):
    """
    construct_answer_prompts():
        Composes and returns a dictionary containing answer-generation prompts 
        for each JSON file in the specified directory and a dictionary of 
        current examples used during prompt composition.

    Args:
        base_dataset_path (str): Path to the directory containing the dataset JSON files.
        per_task_data_row_amount (int): Maximum number of data rows to sample from each task.
        example_source (str): Source of examples, either 'human' or 'llm'.
        prompt_example_dict (dict): Dictionary of previously used examples per task.
        pos_example_amount (int, optional): Number of positive examples to include. Defaults to None.
        neg_example_amount (int, optional): Number of negative examples to include. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - dataset_dict (dict): A dictionary of prompt datasets for each task.
            - current_examples_dict (dict): A dictionary of the current examples used per task.
    """
    # Define an internal helper function to compose example prompts.
    def compose_examples(example_source,
                         task_examples,
                         pos_example_amount,
                         negative_example_amount):
        """
        compose_examples():
            Helper function that concatenates prompts from 
            positive/negative or general examples for a specific task.

        Args:
            example_source (str): Source of examples, 'human' or 'llm'.
            task_examples (dict or list): Examples from either the dataset or previously loaded dictionary.
            pos_example_amount (int): Number of positive examples to include if present.
            negative_example_amount (int): Number of negative examples to include if present.

        Returns:
            tuple: A string containing the concatenated prompt and 
                   a list of example entries used in the prompt.
        """
        # Initialize the prompt as an empty string.
        prompt = ""
        # Initialize a list to hold the chosen examples.
        examples = []
        # Check if the example source is 'human'.
        if example_source == "human":
            # Prepare a local list of examples.
            if pos_example_amount:
                # Slice the positive examples based on the desired amount.
                pos_example = task_examples["Positive Examples"][:pos_example_amount]
                # Add these positive examples to the master list.
                examples += pos_example
                # Construct the prompt snippet for each positive example.
                for each_example in pos_example:
                    prompt += (
                        f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]}\n\n"""
                    )

            # If negative examples are specified, handle them similarly.
            if negative_example_amount:
                neg_example = task_examples["Negative Examples"][:negative_example_amount]
                examples += neg_example
                # Check if the negative examples are valid before constructing the prompt.
                if '-' not in neg_example:
                    for each_example in neg_example:
                        prompt += (
                            f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]}\n\n"""
                        )

        else:
            # If the example source is 'llm', we assume 'task_examples' is already a list.
            examples = task_examples
            # Construct the prompt snippet for each existing example.
            for each_example in task_examples:
                prompt += (
                    f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]}\n\n"""
                )

        # Return the combined prompt and the list of examples used.
        return prompt, examples

    # Initialize the final dataset dictionary.
    dataset_dict = {}
    # Initialize the dictionary to store the current iteration examples.
    current_examples_dict = {}

    # Iterate over each JSON file in the dataset directory.
    for idx, each_json in enumerate(os.listdir(base_dataset_path)):
        # Process only files with a .json extension.
        if ".json" in each_json:
            # Initialize a list to hold prompts for a single task.
            per_task_prompt_list = []
            # Build the full path to the current JSON file.
            full_path = os.path.join(base_dataset_path, each_json)

            # Open and load the JSON file.
            with open(full_path) as obj:
                file = json.loads(obj.read())

            # Create a deep copy to avoid modifying the original file in memory.
            per_task_dict = copy.deepcopy(file)

            # Extract various parts of the dataset: definition, caution, instances, etc.
            task_definition = file["Definition"]
            caution = file["Emphasis & Caution"]
            # Determine how many instances to sample up to 'per_task_data_row_amount'.
            full_length = per_task_data_row_amount
            if full_length > len(file["Instances"]):
                full_length = len(file["Instances"])
            # Slice the instances to the determined length.
            instances = file["Instances"][:full_length]

            # Replace the original instances with the sliced subset.
            per_task_dict["Instances"] = instances

            # If we have any positive or negative examples to include, compose them.
            if pos_example_amount or neg_example_amount:
                if example_source == "human":
                    example_prompt, per_task_examples = compose_examples(
                        example_source,
                        file["Examples"],
                        pos_example_amount,
                        neg_example_amount
                    )
                else:
                    example_prompt, per_task_examples = compose_examples(
                        example_source,
                        prompt_example_dict[each_json],
                        pos_example_amount,
                        neg_example_amount
                    )

            # Construct the instruction portion by combining definition and caution.
            instruction = f"""{task_definition} {caution}\n\n"""
            # Provide a context string for the model.
            context = (
                f"""Please refer to the instruction and task information and """
                f"""give your answers."""
            )
            # If using examples, inform the model that these examples exist.
            if pos_example_amount or neg_example_amount:
                context += "You need to follow the examples we provided."

            # Initialize a placeholder for the examples section.
            example_str = ""
            # If examples are present, incorporate them in a separate section.
            if pos_example_amount or neg_example_amount:
                example_str = f"### Examples:\n{example_prompt}"

            # Construct final prompts for each instance in the dataset.
            for idx, instance in enumerate(instances):
                full_prompt = (
                    f"""### Context:\n{context}\n\n### Instruction:\n{instruction}"""
                    f"""{example_str}### Task:\n{instance['input']}\n\n### Answer:\n"""
                )
                per_task_prompt_list.append(full_prompt)

            # Attach the newly composed prompts to the task dictionary.
            per_task_dict["Answer Prediction Prompt Dataset"] = per_task_prompt_list

            # If we used some examples, store them in both the per-task dict and current_examples_dict.
            if pos_example_amount or neg_example_amount:
                per_task_dict["Current Examples"] = per_task_examples
                current_examples_dict[each_json] = per_task_examples

            # Assign the per-task dictionary to the main dataset.
            dataset_dict[each_json] = per_task_dict
            # Remove the reference to free memory (optional, but sometimes useful).
            del per_task_dict

    # Return both the final dataset dictionary and the current examples used.
    return dataset_dict, current_examples_dict


def construct_feedback_prompts(loaded_examples,
                               answer_pred_dataset):
    """
    construct_feedback_prompts():
        Composes and returns a dictionary of feedback-generation prompts 
        that incorporate model-generated answers (from 'answer_pred_dataset') 
        and optionally some existing examples ('loaded_examples').

    Args:
        loaded_examples (dict): Dictionary of examples that can be used to guide feedback.
        answer_pred_dataset (dict): Dataset of tasks and model predictions.

    Returns:
        dict: A dictionary containing feedback prompt datasets for each task, plus 
              aggregated feedback inputs if needed.
    """
    # Define an internal helper function to build feedback examples in a prompt format.
    def compose_feedback_examples(examples):
        """
        compose_feedback_examples():
            Builds a single concatenated string of feedback examples.

        Args:
            examples (list): List of example entries that contain 
                             input, output, and a reason for correction.

        Returns:
            str: A concatenated string of feedback examples.
        """
        # Start with an empty string for the prompt.
        prompt = ""
        # Loop through each example and construct the snippet.
        for each_example in examples:
            single_prompt = (
                f"""### Task:\n{each_example["input"]}\n\n"""
                f"""### Predicted Answer:\n{each_example["output"]}\n\n"""
                f"""###Feedback:\n{each_example["reason"]}. """
                f"""So the answer should be {each_example["output"]}\n\n"""
            )
            # Concatenate to the overall prompt string.
            prompt += single_prompt

        # Return the fully composed string of feedback examples.
        return prompt

    # Initialize the dictionary that will hold feedback data per task.
    dataset_dict = {}

    # Initialize lists for storing aggregated prompts if needed.
    all_task_feedback_input_data = []
    all_task_feedback_gen_prompt_data = []  # We'll remove the usage if it's not stored.

    # Iterate through each task and its data in the answer prediction dataset.
    for task_name, task_dict in answer_pred_dataset.items():
        # Prepare a list to hold new feedback prompts for this task.
        per_task_prompt_list = []
        # Prepare a list to hold feedback prompts without examples.
        per_task_no_example_input_list = []
        # Deep copy the task dictionary to avoid mutating the original.
        per_task_dict = copy.deepcopy(task_dict)

        # Extract task definition, caution, and instances.
        task_definition = per_task_dict["Definition"]
        caution = per_task_dict["Emphasis & Caution"]
        instances = per_task_dict["Instances"]

        # If we have loaded examples, compose the feedback examples prompt.
        if len(list(loaded_examples.keys())) != 0:
            example_prompt = compose_feedback_examples(loaded_examples[task_name])

        # Provide a feedback prompt guiding the model to analyze predicted answers.
        feedback_prompt = (
            "Please refer to the instruction and task information, "
            "provide your feedback for whether the predict answer is proper, "
            "the reasons and what the correct answer is. "
            "You need to follow the examples we provided."
        )
        # Combine feedback prompt with the actual instruction text for a full context.
        context = f"""{feedback_prompt}\n\n### Instruction:\n{task_definition} {caution} \n\n"""

        # Create a shorter prompt when no examples are available.
        feedback_input_no_examples = (
            "Please refer to the instruction and task information, "
            "provide your feedback for whether the predict answer is proper, "
            "the reasons and what the correct answer is."
        )
        no_example_context = f"""{feedback_input_no_examples}\n\n### Instruction:\n{task_definition} {caution} \n\n"""

        # Build feedback prompts for each instance in the dataset.
        for instance in instances:
            # Extract the instance output, which could be a list in some datasets.
            standard_answer = instance['output']
            if isinstance(standard_answer, list):
                standard_answer = instance['output'][0]

            # Initialize an empty example_str that might hold example prompts.
            example_str = ""
            # If examples exist, add them into the prompt.
            if len(list(loaded_examples.keys())) != 0:
                example_str = f"### Examples:\n{example_prompt}"

            # Construct the full feedback prompt with examples included.
            full_prompt = (
                f"""### Context:\n{context}{example_str}"""
                f"""### Task:\n{instance['input']}\n\n"""
                f"""### Predicted Answer:\n{instance['answer_prediction']}\n\n"""
                f"""### Feedback:\n"""
            )
            # Construct a fallback prompt without examples.
            no_example_full_prompt = (
                f"""### Context:\n{no_example_context}"""
                f"""### Task:\n{instance['input']}\n\n"""
                f"""### Predicted Answer:\n{instance['answer_prediction']}\n\n"""
                f"""### Feedback:\n"""
            )

            # If no examples are loaded, use the no_example prompt list.
            if len(list(loaded_examples.keys())) == 0:
                per_task_prompt_list.append(no_example_full_prompt)
            else:
                # Otherwise, use the full prompt containing examples.
                per_task_prompt_list.append(full_prompt)

            # Keep a copy of the no-example prompts if needed later.
            per_task_no_example_input_list.append(no_example_full_prompt)

        # Store the newly built feedback prompts in the task dictionary.
        per_task_dict["Feedback Prediction Prompt Dataset"] = per_task_prompt_list

        # Assign the per-task dictionary back into the main dataset.
        dataset_dict[task_name] = per_task_dict

        # Add the no-example prompts to the aggregated list.
        all_task_feedback_input_data += per_task_no_example_input_list
        # Add the full prompts to the aggregated list.
        all_task_feedback_gen_prompt_data += per_task_prompt_list

    # Store all feedback input prompts in the dataset dictionary.
    dataset_dict["all_feedback_input_list"] = all_task_feedback_input_data

    # Return the final dictionary of feedback prompts.
    return dataset_dict

