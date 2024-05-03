#!/bin/bash



jobid=$(sbatch /home/qianxi/scratch/laffi/code/scripts/7b_ablation/submit_job_7b_1iter_exp_5mv_no_prompt_opt.sh 5 0 | awk '{print $4}')




# Nested loops for submitting dependent jobs
for i in {1..4}; do
             
    jobid=$(sbatch --dependency=afterok:${jobid} /home/qianxi/scratch/laffi/code/scripts/7b_ablation/submit_job_7b_1iter_exp_5mv_no_prompt_opt.sh 5 "$i" | awk '{print $4}')




done


