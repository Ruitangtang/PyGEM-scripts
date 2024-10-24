#!/bin/bash


step=$1

curdir=$(pwd)



cd $curdir

if [ $step == 1 ]; then
    cd /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808
    cd Output
    rm -rf *
    cd ..
    if ls *.log 1> /dev/null 2>&1; then
        rm *.log
    else
        echo "No .log files to remove."
    fi

    rm -rf oggm_gdirs
    rm -rf calving_data/analysis
    cd $curdir
    echo "Step 1: Run the calibration"
    sed -i "/include_calving/c\include_calving = False" pygem_input.py
    sed -i '0,/^[^#]*hugonnet_fn = 'df_pergla_global_20yr-filled-facorrected.csv'/s//hugonnet_fn = 'df_pergla_global_20yr-filled.csv'/' pygem_input.py
    sed -i '0,/^[^#]*hugonnet_fn = 'df_pergla_global_20yr-filled-facorrected.csv'/s//hugonnet_fn = 'df_pergla_global_20yr-filled.csv'/' pygem_input.py
    sed -i '0,/^[^#]*calving_fp =  main_directory + '/../calving_data/analysis/'/s//calving_fp =  main_directory + '/../calving_data/'/' pygem_input.py
    sed -i '0,/^[^#]*calving_fn = 'all-calving_cal_ind.csv'/s//calving_fn = 'frontalablation_data_test.csv'/' pygem_input.py
    python -u run_calibration.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_01.log > /dev/null
elif [ $step == 2 ]; then
    echo "Step 2: Run the calibration for FA (individual calibration)"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    sed -i "0,/^[^#]*option_ind_calving_k = False/c\option_ind_calving_k = True" run_calibration_FA_Rt_New.py
    sed -i "0,/^[^#]*option_merge_calving_k = True/c\option_merge_calving_k = False" run_calibration_FA_Rt_New.py
    sed -i '0,/^[^#]*option_update_mb_data = True/s//option_update_mb_data = False/' run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_02.log > /dev/null
elif [ $step == 3 ]; then
    echo "Step 3: Run the calibration for FA (merge calving_k for the region)"
    sed -i '0,/^[^#]*option_ind_calving_k = True/s//option_ind_calving_k = False/' run_calibration_FA_Rt_New.py
    sed -i '0,/^[^#]*option_merge_calving_k = False/s//option_merge_calving_k = True/' run_calibration_FA_Rt_New.py
    sed -i '0,/^[^#]*option_update_mb_data = True/s//option_update_mb_data = False/' run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_03.log > /dev/null
elif [ $step == 4 ]; then
    echo "Step 4: Run the calibration for FA (update the geodetic mb - fa )"
    sed -i '0,/^[^#]*option_ind_calving_k = True/s//option_ind_calving_k = False/' run_calibration_FA_Rt_New.py
    sed -i '0,/^[^#]*option_merge_calving_k = True/s//option_merge_calving_k = False/' run_calibration_FA_Rt_New.py
    sed -i '0,/^[^#]*option_update_mb_data = False/s//option_update_mb_data = True/' run_calibration_FA_Rt_New.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_04.log > /dev/null
elif [ $step == 5 ]; then
    echo "Step 5: Update datasets (mb) and recalibrate model parameters"
    sed -i "0,/^[^#]*hugonnet_fn = 'df_pergla_global_20yr-filled.csv'/s//hugonnet_fn = 'df_pergla_global_20yr-filled-facorrected.csv'/" pygem_input.py
    #TODO something went wrong when sed the calving_fp, with the error sed: -e expression #1, char 43: unknown command: `.'
    sed -i "0,/^[^#]*|calving_fp =  main_directory + /../calving_data|'/s//|calving_fp =  |main_directory + /../calving_data/analysis|'/" pygem_input.py
    sed -i "0,/^[^#]*calving_fn = 'frontalablation_data_test.csv'/s//calving_fn = 'all-calving_cal_ind.csv'/" pygem_input.py
    python -u run_calibration.py -debug=1 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_05.log > /dev/null
elif [ $step == 6 ]; then
    echo "Step 6: Run the calibration with mcmc method"
    sed -i "0,/^[^#]*option_calibration = 'emulator'/s|option_calibration = 'emulator'|option_calibration = 'MCMC'|" pygem_input.py
    python -u run_calibration.py -debug=1 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_06.log > /dev/null
elif [$step == 7]; then
    echo "Step 7: Run the simulation for the present day (2000-2020)"
    sed -i "0,/gcm_startyear/s/gcm_startyear.*/gcm_startyear = 2000/" pygem_input.py
    sed -i "0,/gcm_endyear/s/gcm_endyear.*/gcm_endyear = 2019/" pygem_input.py
    sed -i "0,/^[^#]*option_dynamics = None/s/option_dynamics = None.*/option_dynamics ='OGGM'/" pygem_input.py
    python -u run_simulation_FA_Rt.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_07.log > /dev/null
elif [$step == 8]; then
    echo "Step 8: Run the simulation for the future (2000-2100)"
    sed -i "0,/gcm_startyear/s/gcm_startyear.*/gcm_startyear = 2000/" pygem_input.py
    sed -i "0,/gcm_endyear/s/gcm_endyear.*/gcm_endyear = 2100/" pygem_input.py
    sed -i "0,/^[^#]*option_dynamics = None/s/option_dynamics = None.*/option_dynamics ='OGGM'/" pygem_input.py
    sed -i "s/^\([[:space:]]*use_reg_glena = \)True/\1False/" pygem_input.py
    python -u run_simulation_FA_Rt.py -option_parallels=0 -gcm_name='mri-esm2-0' -scenario='ssp126' 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_17.15808/Step_08.log > /dev/null




    



else
    echo "Invalid step"
fi