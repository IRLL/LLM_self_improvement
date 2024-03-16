#!/bin/bash
rm -rf logs/gpu_logs/7b_gpu_usage_lora_4bit_batch8.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_gpu_usage_lora_4bit_batch8.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/qianxi/scratch/laffi/code/finetune.py --enable_ds=0 --model_name="7b" --dataset_path="/home/qianxi/scratch/laffi/code/results/2024-03-14-15-50-53/0/feedback_data.json"

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID