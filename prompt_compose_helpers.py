import json
import os
import copy


def construct_answer_prompts(base_dataset_path,
                             per_task_data_row_amount,
                             example_source,
                             prompt_example_dict,
                             pos_example_amount=None,
                             neg_example_amount=None):
    """
    The goal of this function is to compose the prompts for answer generation.

    for each task:
        1. Accepts (1)the dataset path (2) task examples,
        sample data rows form the full dataset and from different
        tasks.
        2. Compose full prompts for answer generation
        3. Return a dictionary and write a dict to a json file:


    """

    def compose_examples(example_source,
                         task_examples,
                         pos_example_amount,
                         negative_example_amount):
        prompt = ""
        if example_source == "human":
            examples = []
            if pos_example_amount:
                pos_example = task_examples["Positive Examples"][:pos_example_amount]
                examples += pos_example
                for each_example in pos_example:
                    prompt += f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]}\n\n"""

            if negative_example_amount:
                neg_example = task_examples["Negative Examples"][:negative_example_amount]
                examples += neg_example

                if '-' not in neg_example:
                    for each_example in neg_example:
                        prompt += f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]}\n\n"""

        else:
            examples = task_examples
            for each_example in task_examples:
                prompt += f"""### Task:\n{each_example["input"]}\n\n### Answer:\n{each_example["output"]}\n\n"""

        return prompt, examples

    dataset_dict = {}
    current_examples_dict = {}
    for idx, each_json in enumerate(os.listdir(base_dataset_path)):
        if ".json" in each_json:
            per_task_prompt_list = []
            full_path = os.path.join(base_dataset_path, each_json)

            with open(full_path) as obj:
                file = json.loads(obj.read())

            per_task_dict = copy.deepcopy(file)

            task_definition = file["Definition"]
            caution = file["Emphasis & Caution"]
            full_length = per_task_data_row_amount
            if full_length > len(file["Instances"]):
                full_length = len(file["Instances"])
            instances = file["Instances"][:full_length]

            per_task_dict["Instances"] = instances
            if example_source == "human":
                # Compose examples prompt.
                example_prompt, per_task_examples = compose_examples(example_source,
                                                                     file["Examples"],
                                                                     pos_example_amount,
                                                                     neg_example_amount)
            else:
                example_prompt, per_task_examples = compose_examples(example_source,
                                                                     prompt_example_dict[each_json],
                                                                     pos_example_amount,
                                                                     neg_example_amount)

            # Compose instruction.
            instruction = f"""{task_definition} {caution}\n\n"""

            context = f"""Please refer to the instruction and task information and give your answers. You need to follow the examples we provided."""

            # Compose full_prompt for each instance.
            for idx, instance in enumerate(instances):
                full_prompt = f"""### Context:\n{context}\n\n### Instruction:\n{instruction}### Examples:\n{example_prompt}### Task:\n{instance['input']}\n\n### Answer:\n"""
                per_task_prompt_list.append(full_prompt)

            per_task_dict["Answer Prediction Prompt Dataset"] = per_task_prompt_list
            per_task_dict["Current Examples"] = per_task_examples

            current_examples_dict[each_json] = per_task_examples

            dataset_dict[each_json] = per_task_dict
            del per_task_dict

    return dataset_dict, current_examples_dict


def construct_feedback_prompts(loaded_examples,
                               answer_pred_dataset):
    """
The goal of this function is to compose the prompts for feedback generation.

for each task:
    1. Accepts (1)the dataset path (2) task examples,
    sample data rows form the full dataset and from different
    tasks.
    2. Compose full prompts for feedback generation
    3. Return a dictionary and write a dict to a json file:

"""
    def compose_feedback_examples(examples):
        prompt = ""
        for each_example in examples:
            single_prompt = f"""### Task:\n{each_example["input"]}\n\n### Standard Answer:\n{each_example["output"]}\n\n### Predicted Answer:\n{each_example["output"]}\n\n###Feedback:\n{each_example["reason"]}. So the answer should be {each_example["output"]}\n\n"""
            prompt += single_prompt

        return prompt

    dataset_dict = {}
    current_examples_dict = {}

    all_task_feedback_input_data = []
    all_task_feedback_gen_prompt_data = []

    for task_name, task_dict in answer_pred_dataset.items():
        per_task_prompt_list = []
        per_task_no_example_input_list = []
        per_task_dict = copy.deepcopy(task_dict)

        task_definition = per_task_dict["Definition"]
        caution = per_task_dict["Emphasis & Caution"]
        instances = per_task_dict["Instances"]

        # Compose examples prompt.
        example_prompt = compose_feedback_examples(loaded_examples[task_name])

        feedback_prompt = """Please refer to the instruction and task information, compare the standard answer and the predicted answer, provide your feedback for whether you think the predict answer is proper and the reasons. You need to follow the examples we provided."""
        context = f"""{feedback_prompt}\n\n### Instruction:\n{task_definition} {caution} \n\n"""

        feedback_input_no_examples = """Please refer to the instruction and task information, compare the standard answer and the predicted answer, provide your feedback for whether you think the predict answer is proper and the reasons."""
        no_example_context = f"""{feedback_input_no_examples}\n\n### Instruction:\n{task_definition} {caution} \n\n"""

        # Compose full_prompt for each instance.
        for instance in instances:
            standard_answer = instance['output']
            if isinstance(standard_answer, list):
                standard_answer = instance['output'][0]

            full_prompt = f"""### Context:\n{context}### Examples:\n{example_prompt}### Task:\n{instance['input']}\n\n### Standard Answer:\n{standard_answer}\n\n### Predicted Answer:\n{instance['answer_prediction']}\n\n### Feedback:\n"""
            no_example_full_prompt = f"""### Context:\n{no_example_context}### Task:\n{instance['input']}\n\n### Standard Answer:\n{standard_answer}\n\n### Predicted Answer:\n{instance['answer_prediction']}\n\n### Feedback:\n"""

            per_task_prompt_list.append(full_prompt)
            per_task_no_example_input_list.append(no_example_full_prompt)

        per_task_dict["Feedback Prediction Prompt Dataset"] = per_task_prompt_list
        # per_task_dict["Feedback Input Dataset"] = per_task_no_example_input_list

        dataset_dict[task_name] = per_task_dict
        all_task_feedback_input_data += per_task_no_example_input_list
        all_task_feedback_gen_prompt_data += per_task_prompt_list

    dataset_dict["all_feedback_input_list"] = all_task_feedback_input_data
    # dataset_dict["all_task_feedback_gen_prompt_data"] = all_task_feedback_gen_prompt_data
    # Write feedback prompts and data to file.

    return dataset_dict
