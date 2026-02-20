#!/bin/bash
# This script is used to run the calibration and simulation for the  glacier, specifically for frontal ablation paratmeter, tau, using
# the annual timeseries of dL/dt and FA data based on the PBS method.
# This script is a copy of run_calib_dLdt_FA.sh, but is modified to run the calibration and simulation in parallel.
# Author: Ruitang Yang

steps=(1 2 3 4 5 6 7)
rregion=17


curdir=$(pwd)



cd $curdir
if [ -d $curdir/Output ]; then
    path_home="$curdir/Output"
else
    echo "Directory 'Output' does not exist. Creating it."
    mkdir Output
    path_home="$curdir/Output"
fi

path_log1="$path_home/Step_01.log"
path_log2="$path_home/Step_02.log"
path_log3="$path_home/Step_03.log"
path_log4="$path_home/Step_04.log"                                     
path_log5="$path_home/Step_05.log"
path_log6="$path_home/Step_06.log"
path_log6_2="$path_home/Step_06_2.log"
path_log6_3="$path_home/Step_06_3.log"
path_log7="$path_home/Step_07.log"
path_log8_126="$path_home/Step_08_126.log"
path_log8_245="$path_home/Step_08_245.log"
path_log8_370="$path_home/Step_08_370.log"
path_log8_585="$path_home/Step_08_585.log"
output_folder="Output"

echo "Step $step: Run the OGGM pipeline"
#do step in steps[]
if [ $step == 1 ]; then
    cd "$path_home" || { echo "Failed to enter directory $path_home"; exit 1; }
    # Check if Output folder exists, create if not, and change into it
    if [ -d "$output_folder" ]; then
        echo "Directory '$output_folder' exists. Changing into it."
        cd "$output_folder" || { echo "Failed to enter directory $output_folder"; exit 1; }
            # Confirm deletion in Output directory
        read -p "Are you sure you want to delete all files in $path_home/$output_folder? (y/n) " confirm
        if [ "$confirm" = "y" ]; then
            rm -rf *
        fi
        cd ..
    else
        echo "Directory '$output_folder' does not exist. Creating it."
        mkdir "$output_folder"
    fi
    
   
    # Confirm deletion of .log files in $path_home
    if ls *.log 1> /dev/null 2>&1; then
        read -p "Are you sure you want to delete all .log files in $path_home? (y/n) " confirm_log
        if [ "$confirm_log" = "y" ]; then
            rm *.log
        fi
    else
        echo "No .log files to remove."
    fi

    # Confirm deletion of oggm_gdirs and calving_data/analysis directories
    read -p "Are you sure you want to delete oggm_gdirs and calving_data/analysis? (y/n) " confirm_dirs
    if [ "$confirm_dirs" = "y" ]; then
        rm -rf oggm_gdirs
        rm -rf calving_data/analysis
    fi

    cd $curdir
    echo "Step 1: Run the update Geodetic MB by the observed FA (update the geodetic mb - fa)"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    str_glacier_no="glac_no = ['17.04876']"
    sed -i "s:^glac_no =.*$:$str_glacier_no:g" pygem_input.py
    str_main_directory="main_directory = '$path_home/Output/'"
    sed -i "s:^main_directory.*$:$str_main_directory:g" pygem_input.py
    str_include_debris="include_debris = False"
    sed -i "s:^include_debris.*$:$str_include_debris:g" pygem_input.py
    str_hugonnet_fn="hugonnet_fn = 'df_pergla_global_20yr-filled.csv'"
    sed -i "s:^hugonnet_fn.*$:$str_hugonnet_fn:g" pygem_input.py
    str_calving_fp="calving_fp =  main_directory + '/../calving_data/'"
    sed -i "s:^calving_fp.*$:$str_calving_fp:g" pygem_input.py
    str_calving_fn="calving_fn = 'frontalablation_data_test.csv'"
    sed -i "s:^calving_fn.*$:$str_calving_fn:g" pygem_input.py
    str_regions="regions = [${rregion}]"
    sed -i "s:^regions =.*$:$str_regions:g" Update_mb_data_with_FA_individual.py
    #python -u Update_mb_data_with_FA_individual.py
    #base script for step 1 all of the above goes into the base script
    #last line of the base script: srun [options] python -u Update_mb_data_with_FA_individual.py
    sbatch  --job-name=ASD myJob${step}.sh

elif [ $step == 2 ]; then
    echo "Step 2: Run the calibration with calving,using emulator method"
    str_option_calibration="option_calibration = 'emulator'"
    sed -i "s:^option_calibration.*$:$str_option_calibration:g" pygem_input.py
    str_option_dynamics="option_dynamics = 'OGGM'"
    sed -i "s:^option_dynamics.*$:$str_option_dynamics:g" pygem_input.py
    sed -i "s/\(use_reg_glena\s*=\s*\).*/\1True/" ./pygem_input.py
    str_hugonnet_fn="hugonnet_fn = 'df_pergla_global_20yr-filled-facorrected.csv'"
    sed -i "s:^hugonnet_fn.*$:$str_hugonnet_fn:g" pygem_input.py
    python -u run_calibration.py 2>&1 | tee "$path_log2" > /dev/null

elif [ $step == 3 ]; then
    echo "Step 3: Run the calibration with mcmc method"
    str_option_calibration="option_calibration = 'MCMC'"
    sed -i "s:^option_calibration.*$:$str_option_calibration:g" pygem_input.py
    python -u run_calibration.py -debug=1
    #python -u run_calibration.py -debug=1 2>&1 | tee "$path_log6" > /dev/null
    sbatch  --dependency=singlton --after-ok --job-name=ASD myJob${step}.sh

elif [ $step == 4 ]; then
    echo "Step 4: Run the calibration for FA (individual calibration)"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    str_option_ind_calving_k="option_ind_calving_k = True"
    sed -i "s:^option_ind_calving_k.*$:$str_option_ind_calving_k:g" run_calibration_MB_FA_Paralle.py
    str_option_merge_calving_k="option_merge_calving_k = False"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_MB_FA_Paralle.py
    str_option_update_mb_data="option_update_mb_data = False"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_MB_FA_Paralle.py
    python -u run_calibration_MB_FA_Paralle.py
    #python -u run_calibration_MB_FA_Paralle.py 2>&1 | tee "$path_log2" > /dev/null
    sbatch  --dependency=singlton --after-ok --job-name=ASD myJob${step}.sh

elif [ $step == 5 ]; then
    echo "Step 5: Run the calibration for FA (merge calving_k for the region)"
    str_option_ind_calving_k="option_ind_calving_k = False"
    sed -i "s:^option_ind_calving_k.*$:$str_option_ind_calving_k:g" run_calibration_MB_FA_Paralle.py
    str_option_merge_calving_k="option_merge_calving_k = True"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_MB_FA_Paralle.py
    str_option_update_mb_data="option_update_mb_data = False"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_MB_FA_Paralle.py
    python -u run_calibration_MB_FA_Paralle.py 
    #python -u run_calibration_MB_FA_Paralle.py 2>&1 | tee "$path_log3" > /dev/null
    sbatch  --dependency=singlton  --job-name=ASD myJob${step}.sh

elif [ $step == 6 ]; then
    echo "Step 6: Run the simulation for the present day (2000-2020)"
    sed -i "s/\(use_reg_glena\s*=\s*\).*/\1False/" ./pygem_input.py
    str_gcm_startyear="gcm_startyear = 2000"
    sed -i "s:^gcm_startyear.*$:$str_gcm_startyear:g" pygem_input.py
    str_gcm_endyear="gcm_endyear = 2019"
    sed -i "s:^gcm_endyear.*$:$str_gcm_endyear:g" pygem_input.py
    str_option_dynamics="option_dynamics = 'OGGM'"
    sed -i "s:^option_dynamics.*$:$str_option_dynamics:g" pygem_input.py
    str_calving_fp="calving_fp =  main_directory + '/../calving_data/analysis/'"
    sed -i "s:^calving_fp.*$:$str_calving_fp:g" pygem_input.py
    str_calving_fn="calving_fn = 'all-calving_cal_ind.csv'"
    sed -i "s:^calving_fn.*$:$str_calving_fn:g" pygem_input.py
    str_store_monthly_step="store_monthly_step = True"
    sed -i "s:^store_monthly_step.*$:$str_store_monthly_step:g" run_simulation_FA_Rt.py
    str_mb_elev_feedback="mb_elev_feedback = 'monthly'"
    sed -i "s:^mb_elev_feedback.*$:$str_mb_elev_feedback:g" run_simulation_FA_Rt.py
    str_Dynamic_step_Monthly="Dynamic_step_Monthly = True"
    sed -i "s:^Dynamic_step_Monthly.*$:$str_Dynamic_step_Monthly:g" run_simulation_FA_Rt.py
    python -u run_simulation_FA_Rt.py 2>&1 | tee "$path_log7" > /dev/null
elif [ $step == 7 ]; then
    echo "Step 7: Run the simulation for the future (2000-2100)"
    sed -i "s/\(use_reg_glena\s*=\s*\).*/\1False/" ./pygem_input.py
    str_gcm_startyear="gcm_startyear = 2000"
    sed -i "s:^gcm_startyear.*$:$str_gcm_startyear:g" pygem_input.py
    str_gcm_endyear="gcm_endyear = 2100"
    sed -i "s:^gcm_endyear.*$:$str_gcm_endyear:g" pygem_input.py
    str_option_dynamics="option_dynamics = 'OGGM'"
    sed -i "s:^option_dynamics.*$:$str_option_dynamics:g" pygem_input.py
    str_store_monthly_step="store_monthly_step = False"
    sed -i "s:^store_monthly_step.*$:$str_store_monthly_step:g" run_simulation_FA_Rt.py
    str_mb_elev_feedback="mb_elev_feedback = 'annual'"
    sed -i "s:^mb_elev_feedback.*$:$str_mb_elev_feedback:g" run_simulation_FA_Rt.py
    str_Dynamic_step_Monthly="Dynamic_step_Monthly = False"
    sed -i "s:^Dynamic_step_Monthly.*$:$str_Dynamic_step_Monthly:g" run_simulation_FA_Rt.py
    python -u run_simulation_FA_Rt_Future.py -gcm_name='CESM2' -scenario='ssp126' 2>&1 | tee "$path_log8_126" > /dev/null
    python -u run_simulation_FA_Rt_Future.py -gcm_name='CESM2' -scenario='ssp245' 2>&1 | tee "$path_log8_245" > /dev/null
    python -u run_simulation_FA_Rt_Future.py -gcm_name='CESM2' -scenario='ssp370' 2>&1 | tee "$path_log8_370" > /dev/null
    python -u run_simulation_FA_Rt_Future.py -gcm_name='CESM2' -scenario='ssp585' 2>&1 | tee "$path_log8_585" > /dev/null
    
else
    echo "Invalid step"
fi

