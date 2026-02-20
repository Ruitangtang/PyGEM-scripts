#!/bin/bash

SCENARIOS="ssp126 ssp585" # "ssp126 ssp245 ssp370 ssp585"
# Create output directory if it doesn't exist
simdir=../Output//Simulation
if [[ -d $simdir ]]; then
    mkdir -p $simdir
fi
# Start overall timer
START_TIME_ALL=$(date +%s)
# Loop over task files from Task0.txt to TaskN.txt, and run simulation for each GCM with different scenario
GLACIERS=('7.00036') # List of glacier IDs to process ('7.00026' '7.00224' '7.00276' '7.00471',...)
for glac_id in "${GLACIERS[@]}"  # Change 1 glacier to however many glaciers you want to run, or expand to {'7.00026','7.00224','7,00276','7,00471'}
do
    # Log file for the current task_id
    LOGFILE="${simdir}/simulation_Glac${glac_id}_log.txt"
    
    # Initialize or clear log file
    echo "Log file created: $LOGFILE" > "$LOGFILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Running Glac $glac_id..." | tee -a "$LOGFILE"
    # Start timer for the individual task
    START_TIME_TASK=$(date +%s)
    for scenario in $SCENARIOS; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') Running scenario: $scenario" | tee -a "$LOGFILE"
        # Attempt to run the Python script and capture output
        if python -u run_simulation_AMIS_SERMeQ.py \
            -option_parallels \
            -rgi_region01 7 \
            -rgi_glac_number "$glac_id" \
            -gcm_startyear 2000 \
            -gcm_endyear 2100 \
            -gcm_list_fn '../../Input/climate_data/cmip6/gcm_cmip6_list.txt' \
            -scenario "$scenario" \
            -hugonnet_fn "mass_balance_obs_20002010.csv" \
            -debug 2>&1 | tee output_log.txt; then
            # If the command succeeded
            echo "$(date '+%Y-%m-%d %H:%M:%S') Successfully completed scenario: $scenario" | tee -a "$LOGFILE"
        else
            # If the command failed, capture only the traceback information
            echo "$(date '+%Y-%m-%d %H:%M:%S') Error occurred while running scenario: $scenario" | tee -a "$LOGFILE"
            # Filter the output for traceback and the next 10 lines after it
            grep -E "Traceback" -A 10 output_log.txt | tee -a "$LOGFILE"
        fi

        # Clean up: remove output_log.txt whether or not the previous command succeeded
        rm -f output_log.txt  # Use -f to suppress errors if the file doesn't exist 
    done
    # End timer for the individual task
    END_TIME_TASK=$(date +%s)
    DIFF_TIME_TASK=$((END_TIME_TASK - START_TIME_TASK)) # Time difference in seconds

    echo "$(date '+%Y-%m-%d %H:%M:%S') Completed Glac $glac_id." | tee -a "$LOGFILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Start time: $(date -d @$START_TIME_TASK)" | tee -a "$LOGFILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') End time: $(date -d @$END_TIME_TASK)" | tee -a "$LOGFILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Total time for Glac $glac_id: $DIFF_TIME_TASK seconds" | tee -a "$LOGFILE"

    echo "-----------------------------------" | tee -a "$LOGFILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for 10 seconds before starting the next task..." | tee -a "$LOGFILE"
    sleep 10  # Optional: wait for a few seconds before starting the next task
    echo "-----------------------------------" | tee -a "$LOGFILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') Starting next task..." | tee -a "$LOGFILE"
done

# End timer for the overall process
END_TIME_ALL=$(date +%s)
DIFF_TIME_ALL=$((END_TIME_ALL - START_TIME_ALL)) # Time difference in seconds
echo "$(date '+%Y-%m-%d %H:%M:%S') All tasks completed." | tee -a "$LOGFILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') Total time taken: $DIFF_TIME_ALL seconds" | tee -a "$LOGFILE"

# command example to run the sinle task
# Uncomment the following line to run a single task without the loop
# Note: Make sure to replace the task_id with the desired task number (0, 1, 2, ..., N)
# This command is equivalent to running the script for task 0.
# Make sure to adjust the paths and filenames as necessary for your environment.

# Example command to run this bash:
# bash run_simulation_AMIS_Region.sh
# Example command to run this bash at the background:
# nohup bash run_simulation_AMIS_Region.sh &
# Example command to run a single task with specific parameters: ONELINE
# python -u run_simulation_AMIS_SERMeQ.py -option_parallels -rgi_region01 7 -gcm_startyear 2000 -gcm_endyear 2025 -gcm_name='CESM2 BCC-CSM2-MR' -scenario='ssp126' -hugonnet_fn "mass_balance_obs_20002010_task0.csv" -debug
# Example command to run a single glacier with specific parameters (specific gcm): ONELINE
# python -u run_simulation_AMIS_SERMeQ.py -option_parallels -rgi_region01 7 -rgi_glac_number '7.00026'  -gcm_startyear 2000 -gcm_endyear 2100 -gcm_name='BCC-CSM2-MR' -scenario='ssp126' -hugonnet_fn "mass_balance_obs_20002010.csv" -debug