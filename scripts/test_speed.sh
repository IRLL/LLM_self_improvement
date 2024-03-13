#!/bin/bash
rm -rf logs/gpu_logs/13b_gpu_usage_lora_4bit.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/13b_gpu_usage_lora_4bit.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py --model_name="13b" &> logs/program_logs/13b_lora_4bit.log

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID