#!/bin/bash

# Loop over task files from Task0.txt to TaskN.txt
for task_id in {0..11}  # Change 9 to however many tasks you have {0..11}
do
    echo "Running task $task_id..."

    python -u run_calibration_AMIS_MB_FA_20002010_Parallel_Log_New.py \
        -rgi_region01 07 \
        -ref_startyear 2000 \
        -ref_endyear 2019 \
        -frontalablation_fn "frontal_ablation_obs_20002010_task${task_id}.csv" \
        -hugonnet_fn "mass_balance_obs_20002010_task${task_id}.csv" \
        -lengthchange_annual_fn "lengthchange_annual_rgi_region01_7_20002020_task${task_id}.csv" \
        -store_monthly_step \
        -Visualize_Index \
        -v -debug
done

# command example to run the sinle task
# Uncomment the following line to run a single task without the loop
# Note: Make sure to replace the task_id with the desired task number (0, 1, 2, ..., N)
# This command is equivalent to running the script for task 0.
# Make sure to adjust the paths and filenames as necessary for your environment.
# Example command to run a single task:
# python -u run_calibration_AMIS_MB_FA_20002010_Parallel_Log_New.py \
# -rgi_region01 07 -ref_startyear 2000 -ref_endyear 2019 \
# -frontalablation_fn "frontal_ablation_obs_20002010_task0.csv" \
# -hugonnet_fn "mass_balance_obs_20002010_task0.csv" \
# -lengthchange_annual_fn "lengthchange_annual_rgi_region01_7_20002020_task0.csv" \
#-store_monthly_step -Visualize_Index -v -debug

# Example command to run a single task with specific parameters: ONELINE
# python -u run_calibration_AMIS_MB_FA_20002010_Parallel_Log_New.py -rgi_region01 07 -ref_startyear 2000 -ref_endyear 2019 -frontalablation_fn "frontal_ablation_obs_20002010_task0.csv" -hugonnet_fn "mass_balance_obs_20002010_task0.csv" -lengthchange_annual_fn "lengthchange_annual_rgi_region01_7_20002020_task0.csv" -store_monthly_step -Visualize_Index -v -debug