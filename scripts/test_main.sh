#!/bin/bash
rm -rf logs/gpu_logs/7b_main_3.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_main_3.log
    sleep 10  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/main.py --base_dataset_path="/home/qianxi/scratch/laffi/datasets/debug" 2>&1

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID