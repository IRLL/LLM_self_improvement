#!/bin/bash

# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/7b_sft.log
    sleep 10  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
wandb disabled;
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b;

CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/sft.py \
                                --model_path="/home/qianxi/scratch/laffi/models/7b" \
                                --experiment_root_path="/home/qianxi/scratch/laffi/code/results/sft/7b"
                                
                               

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID