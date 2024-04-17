#!/bin/bash

# Declare an array
jobid_list=()

# Loop over sequence numbers
for seq_num in {1..10}; do 
    # Calculate jobid


    if [ "$seq_num" -le 5 ]; then
        # Submit a job and store the job ID
        jobid=$(sbatch /home/qianxi/scratch/laffi/code/scripts/7b_experiments/submit_job_7b_1iter_exp.sh "$seq_num" 0 | awk '{print $4}')
    else
        # Submit a different job and store the job ID
        jobid=$(sbatch /home/qianxi/scratch/laffi/code/scripts/7b_experiments/submit_job_7b_1iter_exp_long.sh "$seq_num" 0 | awk '{print $4}')
    fi

    # Append jobid to the list
    jobid_list+=("$jobid")
done
echo "init array: ${jobid_list[@]}"
# Nested loops for submitting dependent jobs
for i in {1..4}; do
    for seq_num in {1..10}; do 
        # Calculate jobid for dependent jobs
        if [ "$seq_num" -le 5 ]; then
            # Submit a dependent job and update the job ID in the array
             
            jobid=$(sbatch --dependency=afterok:${jobid_list[seq_num-1]} /home/qianxi/scratch/laffi/code/scripts/7b_experiments/submit_job_7b_1iter_exp.sh "$seq_num" "$i" | awk '{print $4}')
        else
            # Submit a different dependent job and update the job ID in the array
            jobid=$(sbatch --dependency=afterok:${jobid_list[seq_num-1]} /home/qianxi/scratch/laffi/code/scripts/7b_experiments/submit_job_7b_1iter_exp_long.sh "$seq_num" "$i" | awk '{print $4}')
            
        fi
        echo "job $jobid depends on ${jobid_list[seq_num-1]}, with seq num ${seq_num} and iter ${i}"
        jobid_list[seq_num-1]="$jobid"
    done
done

# Output the final list of job IDs
echo "Modified array: ${jobid_list[@]}"
