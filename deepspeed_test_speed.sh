#!/bin/bash
rm -rf gpu_logs/gpu_usage.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> gpu_logs/gpu_usage.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 finetune.py --deepspeed deepspeed_config.json

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID