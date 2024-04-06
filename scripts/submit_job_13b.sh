#!/bin/bash
#SBATCH --job-name=LaFFi_13b
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2 
#SBATCH --ntasks-per-node=2
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/qianxi/scratch/laffi/slurm/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023 gcc/12.3 cuda/12.2 arrow/14.0.1 python/3.11.5 ; 

source /home/qianxi/scratch/laffi/march_env/bin/activate;
cd /home/qianxi/scratch/laffi/code/;
wandb offline;
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b;

CUDA_VISIBLE_DEVICES=0,1 python /home/qianxi/scratch/laffi/code/main.py \
                                --base_dataset_path="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train" \
                                --enable_boolq_eval=1 \
                                --enable_squad_eval=1 \
                                --wandb_enabled=1 \
                                --enable_gsm8k_eval=1 \
                                --per_task_data_rows=50 \
                                --experiment_name="13b_exp" \
                                --experiment_root_path="/home/qianxi/scratch/laffi/code/results/13b" \
                                --model_path="/home/qianxi/scratch/laffi/models/13b" \
                                --iteration_amount=5 