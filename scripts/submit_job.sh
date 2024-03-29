#!/bin/bash
#SBATCH --job-name=LaFFi
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4 
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/qianxi/scratch/laffi/slurm/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023 gcc/12.3 cuda/12.2 arrow/14.0.1 python/3.11.5 ; 

source /home/qianxi/scratch/laffi/march_env/bin/activate;
cd /home/qianxi/scratch/laffi/code/;

CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/qianxi/scratch/laffi/code/main.py --base_dataset_path="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train" 2>&1 | tee /home/qianxi/scratch/laffi/code/logs/program_logs/7b.log