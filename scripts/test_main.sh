#!/bin/bash
rm -rf logs/gpu_logs/7b_main.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_main.log
    sleep 30  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/qianxi/scratch/laffi/code/main.py

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID