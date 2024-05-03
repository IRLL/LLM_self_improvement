#!/bin/bash

# Start logging GPU usage in the background
while true; do
    nvidia-smi >> logs/gpu_logs/natural_ins.log
    sleep 2  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
CUDA_VISIBLE_DEVICES=0 python eval_natural_ins.py '{"natural_ins_eval_path":"/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/natural_ins_eval_converted.json","model_path":"/home/qianxi/scratch/laffi/models/7b","inference_batch_size":4}'
# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID