import os,json 
from datetime import datetime
from tqdm import tqdm
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# Format the date and time as a string
task_create_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

experiment_root_path = f"/home/qianxi/scratch/laffi/code/results/{task_create_time_str}"
base_dataset_root = "/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train"
model_path = "/home/qianxi/scratch/laffi/models/7b"
# Create experiment root
os.makedirs(experiment_root_path)

experiment_log_root = os.path.join(experiment_root_path,"program_logs")
os.makedirs(experiment_log_root)

target_iterations=3
current_example_path = os.path.join(experiment_root_path,"current_example.json")

# Loop start.
for iteration_version in tqdm(range(target_iterations)):
    # Create folder for the current iteration.
    cur_iter_root_path = os.path.join(experiment_root_path, str(iteration_version))
    os.makedirs(cur_iter_root_path)

    answer_prompt_path = os.path.join(cur_iter_root_path,"answer_prompt.json")
    feedback_prompt_path = os.path.join(cur_iter_root_path,"feedback_prompt.json")
    answer_dataset_path = os.path.join(cur_iter_root_path,"answer_pred_dataset.json")
    feedback_dataset_path = os.path.join(cur_iter_root_path,"feedback_dataset.json")

    # Start from scratch, use human examples to create 
    # answer generation prompts.
    os.system(f'python construct_answer_prompts.py \
                --base_dataset_path "{base_dataset_root}" \
                --answer_prompt_set_path "{answer_prompt_path}"  \
                --iteration_version {iteration_version} \
                --current_examples_path "{current_example_path}"')

    print(f"Answer generation for iter={iteration_version} started")

    # Generate answer prediction dataset.
    os.system(f'CUDA_VISIBLE_DEVICES=0,1,2,3 python answer_inference.py \
                --model_path "{model_path}"\
                --answer_dataset_path "{answer_dataset_path}" \
                --iteration_version {iteration_version} \
                --current_examples_path "{current_example_path}" \
                --answer_prompt_set_path "{answer_prompt_path}"')

    print(f"Answer generation for iter={iteration_version} finished")

    # Feedback generation prompts.
    os.system(f'python construct_feedback_prompts.py \
                --answer_dataset_path "{answer_dataset_path}" \
                --feedback_prompt_set_path "{feedback_prompt_path}"  \
                --iteration_version {iteration_version} \
                --current_examples_path "{current_example_path}"')

    print(f"Feedback generation for iter={iteration_version} started")
    # Generate feedback dataset.
    os.system(f'CUDA_VISIBLE_DEVICES=0,1,2,3 python feedback_inference.py \
                --model_path "{model_path}"\
                --feedback_dataset_path "{feedback_dataset_path}" \
                --iteration_version {iteration_version} \
                --current_examples_path "{current_example_path}" \
                --feedback_prompt_set_path {feedback_prompt_path}') 

    print(f"Feedback generation for iter={iteration_version} finished")

    shutil.copy(current_example_path, os.path.join(cur_iter_root_path, "current_examples.json"))
    assert 1==2