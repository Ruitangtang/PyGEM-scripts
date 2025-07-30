#!/bin/bash

SCENARIOS="ssp126 ssp585" # "ssp126 ssp245 ssp370 ssp585"
# Create output directory if it doesn't exist
mkdir -p /home/ruitang/OGGM-Ruitang/Results/Test_KS_Regional_1Apr2025/Output/Simulation
# Start overall timer
START_TIME_ALL=$(date +%s)
# Loop over task files from Task0.txt to TaskN.txt, and run simulation for each GCM with different scenario
for task_id in 0  # Change 0 to however many tasks you want to run, or expand to {0..N}
do
    # Log file for the current task_id
    LOGFILE="/home/ruitang/OGGM-Ruitang/Results/Test_KS_Regional_1Apr2025/Output/Simulation/simulation_task${task_id}_log.txt"
    
    # Clear previous log file if it exists
    > "$LOGFILE"

    echo "Running task $task_id..." | tee -a "$LOGFILE"
    for scenario in $SCENARIOS; do
        echo "Running scenario: $scenario" | tee -a "$LOGFILE"
        # Start timer for the individual task
        START_TIME_TASK=$(date +%s)
        # Run the Python script and log both stdout and stderr
        python -u run_simulation_AMIS_MB_FA_RT_MD_Parallel.py \
            -option_parallels \
            -rgi_region01 7 \
            -gcm_startyear 2000 \
            -gcm_endyear 2025 \
            -gcm_list_fn '/home/ruitang/GeoFag_Ruitang/Test_Tidewater/climate_data/cmip6/gcm_cmip6_list.txt' \
            -scenario "$scenario" \
            -hugonnet_fn "mass_balance_obs_20002010_task${task_id}.csv" \
            -debug >> "$LOGFILE" 2>&1
    done
    # End timer for the individual task
    END_TIME_TASK=$(date +%s)
    DIFF_TIME_TASK=$((END_TIME_TASK - START_TIME_TASK)) # Time difference in seconds
    echo "Completed task $task_id." | tee -a "$LOGFILE"
    wait  # Wait for all background processes to finish before starting the next task
    echo "All tasks for task_id $task_id completed." | tee -a "$LOGFILE"
    echo "-----------------------------------" | tee -a "$LOGFILE"
    echo "Waiting for 10 seconds before starting the next task..." | tee -a "$LOGFILE"
    sleep 10  # Optional: wait for a few seconds before starting the next task
    echo "-----------------------------------" | tee -a "$LOGFILE"
    echo "Starting next task..." | tee -a "$LOGFILE"
done

# End timer for the overall process
END_TIME_ALL=$(date +%s)
DIFF_TIME_ALL=$((END_TIME_ALL - START_TIME_ALL)) # Time difference in seconds
echo "All tasks completed." | tee -a "$LOGFILE"
echo "Total time taken: $DIFF_TIME_ALL seconds" | tee -a "$LOGFILE"

# command example to run the sinle task
# Uncomment the following line to run a single task without the loop
# Note: Make sure to replace the task_id with the desired task number (0, 1, 2, ..., N)
# This command is equivalent to running the script for task 0.
# Make sure to adjust the paths and filenames as necessary for your environment.

# Example command to run a single task with specific parameters: ONELINE
# python -u run_simulation_AMIS_MB_FA_RT_MD_Parallel.py -option_parallels -rgi_region01 7 -gcm_startyear 2000 -gcm_endyear 2025 -gcm_name='CESM2 BCC-CSM2-MR' -scenario='ssp126' -hugonnet_fn "mass_balance_obs_20002010_task0.csv" -debug