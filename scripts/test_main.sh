#!/bin/bash
rm -rf logs/gpu_logs/7b_test_fb_inf.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_test_fb_inf.log
    sleep 3  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!
current_datetime=$(date "+%Y-%m-%d_%H-%M-%S")
filename="filename_${current_datetime}"
exp_root="/home/qianxi/scratch/laffi/code/results/test2/${filename}"

# for i in {0..2};
# do 
#     CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/main.py \
#                                     --base_dataset_path="/home/qianxi/scratch/laffi/datasets/debug" \
#                                     --enable_boolq_eval=1 \
#                                     --enable_squad_eval=0 \
#                                     --enable_gsm8k_eval=1 \
#                                     --experiment_root_path=$exp_root \
#                                     --per_task_data_rows=5 \
#                                     --cur_iteration=$i
exp_root="/home/qianxi/scratch/laffi/code/results/test2/7b_full_third"
# done
# CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/main.py \
#                                 --base_dataset_path="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train" \
#                                 --enable_boolq_eval=0 \
#                                 --enable_squad_eval=0 \
#                                 --enable_gsm8k_eval=0 \
#                                 --experiment_root_path=$exp_root \
#                                 --per_task_data_rows=50 \
#                                 --cur_iteration=0

# CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/main.py \
#                                 --base_dataset_path="/home/qianxi/scratch/laffi/datasets/debug" \
#                                 --enable_boolq_eval=1 \
#                                 --enable_squad_eval=0 \
#                                 --enable_gsm8k_eval=1 \
#                                 --experiment_root_path=$exp_root \
#                                 --per_task_data_rows=5 \
#                                 --cur_iteration=1
CUDA_VISIBLE_DEVICES=0 python eval_boolq.py '{"cur_iteration":0, "boolq_eval_result_path":"/home/qianxi/scratch/laffi/code/results/test2/7b_full_third/1/boolq_test.json","boolq_eval_path":"/home/qianxi/scratch/laffi/datasets/boolq/eval_boolq.json","adapters_path":"/home/qianxi/scratch/laffi/code/results/test2/7b_full_third/adapters","model_path":"/home/qianxi/scratch/laffi/models/7b"}'
# CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/feedback_inference.py '{"cur_iteration":0, "contamination":0.3,"num_return_seq":5, "adapters_path":"/home/qianxi/scratch/laffi/code/results/test2/7b_full/adapters", "feedback_prompts_path":"/home/qianxi/scratch/laffi/code/results/test2/7b_full/0/feedback_prompts.json", "feedback_dataset_path":"/home/qianxi/scratch/laffi/code/results/test2/7b_full/0/fb_test123.json","current_prompt_examples_path":"/home/qianxi/scratch/laffi/code/results/official_experiment/7b/1_run/0/prompt_examples.json","major_voting_save_path":"/home/qianxi/scratch/laffi/code/results/official_experiment/7b/1_run/0/mv.json","new_example_indices_dict_path":"/home/qianxi/scratch/laffi/code/results/test2/7b_full/0/new_example_indices_dict.json","model_path":"/home/qianxi/scratch/laffi/models/7b"}'

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID