#!/bin/bash
# rm -rf logs/gpu_logs/lmsi_7b_main_1.log
# Start logging GPU usage in the background
while true; do
    nvidia-smi >> lmsi_7b_main_1.log
    sleep 10  # Adjust this value as needed
done &

# Remember the PID of the background process
NVIDIASMI_PID=$!

# Run your Python program
wandb disabled;
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b;

CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/reproducelmsi/lmsi.py \
                                --base_dataset_path="/home/qianxi/scratch/laffi/datasets/debug" \
                                --enable_boolq_eval=0 \
                                --enable_squad_eval=0 \
                                --enable_gsm8k_eval=0 \
                                --per_task_data_rows=5 \
                                --experiment_name="reproduce_lmsi" \
                                --wandb_enabled=0 \
                                --experiment_root_path="/home/qianxi/scratch/laffi/code/reproducelmsi/results/test" \
                                --iteration_amount=2

# Your Python program has finished; now kill the background nvidia-smi logging process
kill $NVIDIASMI_PID