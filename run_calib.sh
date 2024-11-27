#!/bin/bash


step=$1

curdir=$(pwd)



cd $curdir
path_home="/home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808_Test04"
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

echo "Step $step: Run the OGGM pipeline"
if [ $step == 1 ]; then
    cd "$path_home" || { echo "Failed to enter directory $path_home"; exit 1; }
    cd Output || { echo "Output directory not found"; exit 1; }

    # Confirm deletion in Output directory
    read -p "Are you sure you want to delete all files in $path_home/Output? (y/n) " confirm
    if [ "$confirm" = "y" ]; then
        rm -rf *
    fi

    cd ..
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
    echo "Step 1: Run the calibration"
    sed -i "/include_calving/c\include_calving = False" pygem_input.py
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
    str_option_calibration="option_calibration = 'emulator'"
    sed -i "s:^option_calibration.*$:$str_option_calibration:g" pygem_input.py
    str_option_dynamics="option_dynamics = 'OGGM'"
    sed -i "s:^option_dynamics.*$:$str_option_dynamics:g" pygem_input.py
    sed -i "s/\(use_reg_glena\s*=\s*\).*/\1True/" ./pygem_input.py
    python -u run_calibration.py 2>&1 | tee "$path_log1" > /dev/null

elif [ $step == 2 ]; then
    echo "Step 2: Run the calibration for FA (individual calibration)"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    str_option_ind_calving_k="option_ind_calving_k = True"
    sed -i "s:^option_ind_calving_k.*$:$str_option_ind_calving_k:g" run_calibration_FA_Rt_New.py
    str_option_merge_calving_k="option_merge_calving_k = False"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_FA_Rt_New.py
    str_option_update_mb_data="option_update_mb_data = False"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_FA_Rt_New.py
    #python -u run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee "$path_log2" > /dev/null

elif [ $step == 3 ]; then
    echo "Step 3: Run the calibration for FA (merge calving_k for the region)"
    str_option_ind_calving_k="option_ind_calving_k = False"
    sed -i "s:^option_ind_calving_k.*$:$str_option_ind_calving_k:g" run_calibration_FA_Rt_New.py
    str_option_merge_calving_k="option_merge_calving_k = True"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_FA_Rt_New.py
    str_option_update_mb_data="option_update_mb_data = False"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee "$path_log3" > /dev/null
elif [ $step == 4 ]; then
    echo "Step 4: Run the calibration for FA (update the geodetic mb - fa )"
    str_option_merge_calving_k="option_merge_calving_k = False"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_FA_Rt_New.py
    str_option_update_mb_data="option_update_mb_data = True"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee "$path_log4" > /dev/null
elif [ $step == 5 ]; then
    echo "Step 5: Update datasets (mb) and recalibrate model parameters"
    str_hugonnet_fn="hugonnet_fn = 'df_pergla_global_20yr-filled-facorrected.csv'"
    sed -i "s:^hugonnet_fn.*$:$str_hugonnet_fn:g" pygem_input.py
    str_calving_fn="calving_fn = 'all-calving_cal_ind.csv'"
    sed -i "s:^calving_fn.*$:$str_calving_fn:g" pygem_input.py
    str_calving_fp="calving_fp =  main_directory + '/../calving_data/analysis/'"
    sed -i "s:^calving_fp.*$:$str_calving_fp:g" pygem_input.py
    python -u run_calibration.py 2>&1 | tee "$path_log5" > /dev/null
elif [ $step == 6 ]; then
    echo "Step 6: Run the calibration with mcmc method"
    str_option_calibration="option_calibration = 'MCMC'"
    sed -i "s:^option_calibration.*$:$str_option_calibration:g" pygem_input.py
    python -u run_calibration.py -debug=1 2>&1 | tee "$path_log6" > /dev/null
elif [ $step == 6-2 ]; then
    echo "Step 6-2: Run the calibration for FA (individual calibration) after MCMC calibration"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    str_option_ind_calving_k="option_ind_calving_k = True"
    sed -i "s:^option_ind_calving_k.*$:$str_option_ind_calving_k:g" run_calibration_FA_Rt_New_01.py
    str_option_merge_calving_k="option_merge_calving_k = False"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_FA_Rt_New_01.py
    str_option_update_mb_data="option_update_mb_data = False"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_FA_Rt_New_01.py
    #python -u run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New_01.py 2>&1 | tee "$path_log6_2" > /dev/null
elif [ $step == 6-3 ]; then
    echo "Step 6-2: Run the calibration for FA (individual calibration) after MCMC calibration"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    str_option_ind_calving_k="option_ind_calving_k = False"
    sed -i "s:^option_ind_calving_k.*$:$str_option_ind_calving_k:g" run_calibration_FA_Rt_New_01.py
    str_option_merge_calving_k="option_merge_calving_k = True"
    sed -i "s:^option_merge_calving_k.*$:$str_option_merge_calving_k:g" run_calibration_FA_Rt_New_01.py
    str_option_update_mb_data="option_update_mb_data = False"
    sed -i "s:^option_update_mb_data.*$:$str_option_update_mb_data:g" run_calibration_FA_Rt_New_01.py
    #python -u run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New_01.py 2>&1 | tee "$path_log6_3" > /dev/null

elif [ $step == 7 ]; then
    echo "Step 7: Run the simulation for the present day (2000-2020)"
    sed -i "s/\(use_reg_glena\s*=\s*\).*/\1False/" ./pygem_input.py
    str_gcm_startyear="gcm_startyear = 2000"
    sed -i "s:^gcm_startyear.*$:$str_gcm_startyear:g" pygem_input.py
    str_gcm_endyear="gcm_endyear = 2019"
    sed -i "s:^gcm_endyear.*$:$str_gcm_endyear:g" pygem_input.py
    str_option_dynamics="option_dynamics = 'OGGM'"
    sed -i "s:^option_dynamics.*$:$str_option_dynamics:g" pygem_input.py
    str_store_monthly_step="store_monthly_step = True"
    sed -i "s:^store_monthly_step.*$:$str_store_monthly_step:g" run_simulation_FA_Rt.py
    str_mb_elev_feedback="mb_elev_feedback = 'monthly'"
    sed -i "s:^mb_elev_feedback.*$:$str_mb_elev_feedback:g" run_simulation_FA_Rt.py
    str_Dynamic_step_Monthly="Dynamic_step_Monthly = True"
    sed -i "s:^Dynamic_step_Monthly.*$:$str_Dynamic_step_Monthly:g" run_simulation_FA_Rt.py
    python -u run_simulation_FA_Rt.py 2>&1 | tee "$path_log7" > /dev/null
elif [ $step == 8 ]; then
    echo "Step 8: Run the simulation for the future (2000-2100)"
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

