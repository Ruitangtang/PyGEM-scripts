#!/bin/bash

# Loop over task files from Task0.txt to TaskN.txt
for task_id in {1..2}  # Change 9 to however many tasks you have
do
    echo "Running task $task_id..."

    python -u run_calibration_AMIS_MB_FA_20002010_Parallel_Log.py \
        -rgi_region01 07 \
        -ref_startyear 2000 \
        -ref_endyear 2019 \
        -frontalablation_fn "frontal_ablation_obs_20002010_task${task_id}.csv" \
        -hugonnet_fn "mass_balance_obs_20002010_task${task_id}.csv" \
        -lengthchange_annual_fn "lengthchange_annual_rgi_region01_7_20002020_task${task_id}.csv" \
        -store_monthly_step \
        -Visualize_Index \
        -v -o -debug
done
