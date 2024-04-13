


# for i in {0..4};
# do 

jobid=$(sbatch /home/qianxi/scratch/laffi/code/scripts/submit_job_7b_1iter.sh 0 | awk '{print $4}')
echo "Submitted the first job: $jobid"

for i in {1..10}
do
    jobid=$(sbatch --dependency=afterok:$jobid /home/qianxi/scratch/laffi/code/scripts/submit_job_7b_1iter.sh $i | awk '{print $4}')
    echo "Submitted job $i with dependency on job $jobid"
done
