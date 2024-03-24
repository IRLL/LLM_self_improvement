#!/bin/bash
# rm -rf logs/gpu_logs/7b_gpu_usage_inference_4bit.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_squad_gpu_usage_inference_4bit.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/qianxi/scratch/laffi/code/squad_evaluation.py \
--original_squad_eval_set_path='/home/qianxi/scratch/laffi/datasets/squad2/squad_official_eval.json' \
--squad_response_gen_file='/home/qianxi/scratch/laffi/code/results/tmp/squad2.json' \
--squad_eval_result_path='/home/qianxi/scratch/laffi/code/results/tmp/metrics.json' \
--transformed_squad_eval_set_path='/home/qianxi/scratch/laffi/datasets/squad2/processed_eval_dataset.json' \
--model_path='/home/qianxi/scratch/laffi/models/7b'

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID