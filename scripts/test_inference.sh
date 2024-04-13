#!/bin/bash
rm -rf logs/gpu_logs/7b_gpu_usage_inference_4bit.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_gpu_usage_inference_4bit.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0 python mii_test.py

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID