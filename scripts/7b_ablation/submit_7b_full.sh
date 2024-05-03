#!/bin/bash



jobid=$(sbatch /home/qianxi/scratch/laffi/code/scripts/7b_ablation/submit_job_7b_1iter_exp.sh 1 0 | awk '{print $4}')




# Nested loops for submitting dependent jobs
for i in {1..4}; do
             
    jobid=$(sbatch --dependency=afterok:${jobid} /home/qianxi/scratch/laffi/code/scripts/7b_ablation/submit_job_7b_1iter_exp.sh 1 "$i" | awk '{print $4}')
    



done


