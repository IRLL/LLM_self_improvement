import json,os,copy
from utils import parse_arguments

"""
The goal of this file is to compose the prompts for answer generation.

for each task:
    1. Accepts (1)the dataset path (2) task examples,
    sample data rows form the full dataset and from different
    tasks.
    2. Compose full prompts for answer generation
    3. Return a dictionary and write a dict to a json file:


"""

# Load arguments.
args = parse_arguments()

example_source = "human"

# Need to carefully design this:
per_task_data_row_amount = 2

def compose_examples(example_source,
                     human_examples,
                     pos_example_amount,
                     negative_example_amount):
    prompt = "Examples: \n"
    if example_source == "human":
        examples = []
        if pos_example_amount:
            pos_example = human_examples["Positive Examples"][:pos_example_amount]
            examples += pos_example
            for each_example in pos_example:
                single_prompt = f"""Input: {each_example["input"]} \nOutput:{each_example["output"]}\n\n"""
                prompt += single_prompt
            
        if negative_example_amount:
            neg_example = human_examples["Negative Examples"][:negative_example_amount]
            examples += neg_example

            if '-' not in neg_example:
                for each_example in neg_example:
                    single_prompt = f"""Input: {each_example["input"]} \nOutput:{each_example["output"]}\n\n"""
                    prompt += single_prompt

    return prompt, examples

dataset_dict = {}
current_examples_dict = {}
for idx, each_json in enumerate(os.listdir(args.base_dataset_path)):
    if ".json" in each_json:
        per_task_prompt_list = []
        full_path = os.path.join(args.base_dataset_path, each_json)

        with open(full_path) as obj:
            file = json.loads(obj.read())
        
        per_task_dict = copy.deepcopy(file)

        task_definition = file["Definition"]
        caution = file["Emphasis & Caution"]
        instances = file["Instances"][:per_task_data_row_amount]

        per_task_dict["Instances"] = instances
        human_examples = file["Examples"]

        # Compose examples prompt.
        example_prompt, per_task_examples = compose_examples(example_source,
                                          human_examples,
                                          args.pos_example_amount,
                                          args.neg_example_amount)

        # Compose context.
        context = f"""{task_definition} {caution}"""

        # Compose full_prompt for each instance.
        for idx,instance in enumerate(instances):
            full_prompt = f"""{context}\n\n{example_prompt}\n\n###\n\nInput: {instance['input']} \nOutput:"""
            per_task_prompt_list.append(full_prompt)

            # To be removed later
            per_task_dict["Instances"][idx]["answer_prediction"] = "hahahhah"
            

        per_task_dict["Answer Prediction Prompt Dataset"] = per_task_prompt_list
        per_task_dict["Current Examples"] = per_task_examples


        current_examples_dict[each_json] = per_task_examples

        dataset_dict[each_json] = per_task_dict
        

# Write answer prompts and data to file.
with open(args.answer_prompt_set_path,'w') as obj:
    obj.write(json.dumps(dataset_dict))

# Write current examples for each task to a file.
with open(args.current_examples_path,'w') as obj:
    obj.write(json.dumps(current_examples_dict))
