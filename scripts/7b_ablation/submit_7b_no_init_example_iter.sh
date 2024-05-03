#!/bin/bash
#SBATCH --job-name=no_init_exam
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=30:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/qianxi/scratch/laffi/slurm/ablation/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=FAIL,END

module load StdEnv/2023 gcc/12.3 cuda/12.2 arrow/14.0.1 python/3.11.5 ; 

source /home/qianxi/scratch/laffi/march_env/bin/activate;
cd /home/qianxi/scratch/laffi/code/;

echo "sweep ${1} ${2}"
CUDA_VISIBLE_DEVICES=0 python /home/qianxi/scratch/laffi/code/main.py \
                                --base_dataset_path="/home/qianxi/scratch/laffi/datasets/natural_instruction_v1/train" \
                                --enable_boolq_eval=0 \
                                --enable_squad_eval=0 \
                                --enable_gsm8k_eval=1 \
                                --num_return_seq=$1 \
                                --enable_initial_human_examples=0 \
                                --experiment_root_path="/home/qianxi/scratch/laffi/code/results/official_experiment/7b/ablation/no_init_example" \
                                --per_task_data_rows=30 \
                                --cur_iteration=$2 \
                                --eval_inference_batch_size=1
