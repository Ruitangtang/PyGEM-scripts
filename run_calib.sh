#!/bin/bash


step=$1

curdir=$(pwd)



cd $curdir

if [ $step == 1 ]; then
    cd /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_1.10689
    cd Output
    rm -rf *
    cd ..
    rm *.log
    rm -rf oggm_gdirs
    rm -rf calving_data/analysis
    cd $curdir
    echo "Step 1: Run the calibration"
    sed -i "/include_calving/c\include_calving = False" pygem_input.py
    python -u run_calibration.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_1.10689/Step_01.log > /dev/null
elif [ $step == 2 ]; then
    echo "Step 2: Run the calibration"
    sed -i "/include_calving/c\include_calving = True" pygem_input.py
    python -u run_calibration_FA_Rt_New.py 2>&1 | tee /home/ruitang/OGGM-Ruitang/Results/Test_KS_1T_24Jun/RGI_1.10689/Step_02.log > /dev/null
else
    echo "Invalid step"
fi