#!/bin/bash
# rm -rf gpu_logs/gpu_usage_plain.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> gpu_logs/gpu_usage_lora_4bit.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_plain.py &> log/lora_4bit.log

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID