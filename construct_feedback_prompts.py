import json,os,copy
from utils import parse_arguments

"""
The goal of this file is to compose the prompts for feedback generation.

for each task:
    1. Accepts (1)the dataset path (2) task examples,
    sample data rows form the full dataset and from different
    tasks.
    2. Compose full prompts for feedback generation
    3. Return a dictionary and write a dict to a json file:

"""


# Load arguments.
args = parse_arguments()

def compose_examples(examples):
    prompt = "Examples: \n"
    for each_example in examples:
        single_prompt = f"""Input: {each_example["input"]} \nPredicted Answer: {each_example["output"]} \nFeedback: {each_example["reason"]} \n\n"""
        prompt += single_prompt

    return prompt

dataset_dict = {}
current_examples_dict = {}
if os.path.exists(args.current_examples_path):
    with open(args.current_examples_path) as obj:
        loaded_examples = json.loads(obj.read())

with open(args.answer_dataset_path) as obj:
    answer_pred_dataset = json.loads(obj.read())

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
    example_prompt = compose_examples(loaded_examples[task_name])

    feedback_prompt = """Please refer to the context, input, compare the standard answer and the predicted answer, provide your feedback for whether you think the predict answer is proper and the reasons. You need to follow the examples we provided."""
    context = f"""{feedback_prompt}\n\nTask:{task_definition} {caution} \n\n"""

    feedback_input_no_examples = """Please refer to the context, input, compare the standard answer and the predicted answer, provide your feedback for whether you think the predict answer is proper and the reasons."""
    no_example_context = f"""{feedback_input_no_examples}\n\nTask:{task_definition} {caution} \n\n"""

    # Compose full_prompt for each instance.
    for instance in instances:
        standard_answer = instance['output']
        if isinstance(standard_answer, list):
            standard_answer = instance['output'][0]

        full_prompt = f"""{context}\n\n{example_prompt}\n\n###\n\nInput: {instance['input']} \nStandard Answer: {standard_answer} \nPredicted Answer: {instance['answer_prediction']} \n Feedback: """
        no_example_full_prompt = f"""{no_example_context}\n\nInput: {instance['input']} \nStandard Answer: {standard_answer} \nPredicted Answer: {instance['answer_prediction']} \n Feedback: """

        per_task_prompt_list.append(full_prompt)
        per_task_no_example_input_list.append(no_example_full_prompt)

    per_task_dict["Feedback Prediction Prompt Dataset"] = per_task_prompt_list
    per_task_dict["Feedback Input Dataset"] = per_task_no_example_input_list

    dataset_dict[task_name] = per_task_dict
    all_task_feedback_input_data += per_task_no_example_input_list
    all_task_feedback_gen_prompt_data += per_task_prompt_list

dataset_dict["all_feedback_input_list"] = all_task_feedback_input_data
dataset_dict["all_task_feedback_gen_prompt_data"] = all_task_feedback_gen_prompt_data
# Write feedback prompts and data to file.
with open(args.feedback_prompt_set_path,'w') as obj:
    obj.write(json.dumps(dataset_dict))

