"""
Analysis of the calibration output, comparing the predicted values with the true/observation values.
@Author: Ruitang Yang
@Date: 2025.03.12
@Version: 1.0

"""

# Load the build-in libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import ast
import csv  # Import the standard csv module
from scipy.stats import linregress
import argparse
# Load the local libraries
import pygem_input as pygem_prms

import Visualization_timeseries as Vis_ts
import statistic_tool as stats_t
# debug lib
import pdb



#%% plot the timeseries of the calibration output
# ==== plot the length/length change timeseries  vs observations ====

# Read the RGIID list
#rgiid_list = pd.read_csv(pygem_prms.rgiid_list_fp)
#Vis_ts.plot_length_TS_Annual(rgiid='RGI60-17.15808')
#Vis_ts.plot_length_TS_Annual_New(rgiid='RGI60-17.15808')
#is_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808')

# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N200/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N40_New_Th30/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N40_New_Th5/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N40_New_Th10/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N400_New_Th40/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002020_N40/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002020_N200/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20102020_N40/')
# Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20102020_N200/')
#Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N40_New_Th15/')
#Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N1000_New_Th15/')
#Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N400_New_Th50/')
#Vis_ts.plot_length_TS_Annual_New3(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N400_New_Th10/')
#Vis_ts.plot_length_dl_TS_Annual(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N200/')


# ---- Functions ----

# Function to get the arguments from the command line
def getparse():
    """Parse command line arguments.
    Parameters
    ----------
    model_output_fp (optional) : str
        Path to the model output file.
    model_parameters_fp (optional) : str
        Path to the model parameters file.
    obs_data_fp (optional) : str
        Path to the observation data file.
    region_id : int
        Region ID for the analysis.
    data_index : str
        Data index for the analysis. e.g., 'Annual', 'Monthly', etc.
    log_level : str
        Logging level for the analysis. Default is 'INFO', e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    Returns
    -------
    Object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Analysis of the calibration output and validation.')
    # Add arguments to the parser
    parser.add_argument('--model_output_fp', type=str, required= False,
                        help='Path to the model output file.')
    parser.add_argument('--model_parameters_fp', type=str, required= False,
                        help='Path to the model parameters file.')
    parser.add_argument('--obs_data_fp', type=str, required= False,
                        help='Path to the observation data file.')
    # Add required arguments
    parser.add_argument('--region_id', type=int, required=True,
                        help='Region ID for the analysis.')
    parser.add_argument('--data_index', type=str, required=True,
                        help='Data index for the analysis. e.g., "Annual", "Monthly", etc.')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Logging level for the analysis. Default is "INFO". e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".')
    return parser.parse_args()


#%% Plot the Calibration/validation of modeled results (2000-2010) vs observations  (2010-2020)  of 70 glaciers in the RGI region 7, Svalbard , dL, MB_clim, FA
def main ():
    # Get the arguments from the command line
    args = getparse()
    
    # Check if the required arguments are provided    
    if args.region_id is None:
        args.region_id = int(input("Please enter the Region ID: e.g., 7 for Svalbard: "))
    if args.data_index is None:
        args.data_index = input("Please enter the Data Index (e.g., 'Annual', 'Monthly'): ")
    # Set the logging level based on the argument
    log_level = args.log_level.upper()

    # extract the region ID from the arguments
    region_id = args.region_id
    reg_id = str(region_id).zfill(2)  # Ensure region ID is two digits, e.g., '07' for region 7
    # extract the data index from the arguments
    data_index = args.data_index

    # Check if the model output file path is provided
    if not args.model_output_fp:
        # If not provided, use the default model output file path
        model_output_fp = pygem_prms.main_directory + '/Calibration/modeloutput'
    else:
        # If provided, use the model output file path from the arguments
        model_output_fp = args.model_output_fp
    # Check if the model parameters file path is provided
    if not args.model_parameters_fp:
        # If not provided, use the default model parameters file path
        model_parameters_fp = pygem_prms.main_directory + '/Calibration/parameter'
    else:
        # If provided, use the model parameters file path from the arguments
        model_parameters_fp = args.model_parameters_fp
    # Check if the observation data file path is provided
    if not args.obs_data_fp:
        # If not provided, use the default observation data file path
        obs_data_fp = pygem_prms.main_directory + '/../Calibration_dataset/RGI_fit_Obs_fa_dLdt_MB'
    else:
        # If provided, use the observation data file path from the arguments
        obs_data_fp = args.obs_data_fp

    # generate the regional output and parameters file paths
    model_output_fp_region = os.path.join(model_output_fp, reg_id)
    model_param_fp_region = os.path.join(model_parameters_fp, reg_id)

    # generate the regional postprocessing output file path
    postpro_output_fp_region = os.path.join(model_output_fp,'..','Postprocessing', reg_id)
    # create the postprocessing output directory if it does not exist
    if not os.path.exists(postpro_output_fp_region):
        os.makedirs(postpro_output_fp_region)

    # get the AMIS INFO PATH
    AMIS_statis_fp_region = os.path.join(model_output_fp,'..','Statistics_model_run',reg_id)

    # Get the path of AMIS Info of each iteration
    model_AMIS_fp = os.path.join(model_output_fp,'..','AMIS_info')
    model_AMIS_fp_region = os.path.join(model_AMIS_fp,reg_id)

    # == Step 1 : Load the model output statistic info data ====
    all_glac_data_stats,mean_data,sum_data = stats_t.read_extract_data_region(region_output_path = model_output_fp_region,
                                                                            region_params_path= model_param_fp_region,reg_id= reg_id,
                                                                            data_index = 'Annual')
    #pdb.set_trace()
    # == Step 2 : Load the model output raw data ====
    # the total length change over the period 2000-2020,with all glaceries in the region, for each glacier including all the emsemble members
    sum_dL_SQ_0020_raw_m = stats_t.extract_and_save_ensemble_data(sum_data, save_path = postpro_output_fp_region,
                                                                  key_value='lengthchange_dLdt_model_array_annual_myr',
                                                                  file_name='sum_dL_SQ_0020_raw_m',period='2000_2020')
    sum_dL_SQ_1020_raw_m = stats_t.extract_and_save_ensemble_data(sum_data, save_path = postpro_output_fp_region,
                                                                  key_value='lengthchange_dLdt_model_array_annual_myr',
                                                                  file_name='sum_dL_SQ_1020_raw_m',period='2010_2020')
    sum_dL_SQ_0010_raw_m = stats_t.extract_and_save_ensemble_data(sum_data, save_path = postpro_output_fp_region,
                                                                    key_value='lengthchange_dLdt_model_array_annual_myr',
                                                                    file_name='sum_dL_SQ_0010_raw_m',period='2000_2010')
    mean_MB_clim_1020_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_mwea',
                                                                       file_name='mean_MB_clim_1020_raw_mwea',period='2010_2020')
    mean_MB_clim_0020_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_mwea',
                                                                       file_name='mean_MB_clim_0020_raw_mwea',period='2000_2020')
    mean_MB_clim_0010_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_mwea',
                                                                       file_name='mean_MB_clim_0010_raw_mwea',period='2000_2010')
    mean_MB_clim_1020_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_gta',
                                                                       file_name='mean_MB_clim_1020_raw_gta',period='2010_2020')
    mean_MB_clim_0020_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_gta',
                                                                       file_name='mean_MB_clim_0020_raw_gta',period='2000_2020')
    mean_MB_clim_0010_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_gta',
                                                                       file_name='mean_MB_clim_0010_raw_gta',period='2000_2010')
    mean_MB_total_1020_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_mwea',
                                                                       file_name='mean_MB_total_1020_raw_mwea',period='2010_2020')
    mean_MB_total_0020_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_mwea',
                                                                       file_name='mean_MB_total_0020_raw_mwea',period='2000_2020')
    mean_MB_total_0010_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_mwea',
                                                                       file_name='mean_MB_total_0010_raw_mwea',period='2000_2010')
    mean_MB_total_1020_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_gta',
                                                                       file_name='mean_MB_total_1020_raw_gta',period='2010_2020')
    mean_MB_total_0020_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_gta',
                                                                       file_name='mean_MB_total_0020_raw_gta',period='2000_2020')
    mean_MB_total_0010_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_gta',
                                                                       file_name='mean_MB_total_0010_raw_gta',period='2000_2010')
    mean_FA_1020_raw_Gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                  key_value='calving_flux_Gta_TMS_model_array_annual',
                                                                  file_name='mean_FA_1020_raw_Gta',period='2010_2020')
    mean_FA_0010_raw_Gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                  key_value='calving_flux_Gta_TMS_model_array_annual',
                                                                  file_name='mean_FA_0010_raw_Gta',period='2000_2010')
    mean_FA_0020_raw_Gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                    key_value='calving_flux_Gta_TMS_model_array_annual',
                                                                    file_name='mean_FA_0020_raw_Gta',period='2000_2020')
    mean_FA_0020_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                   key_value='FA_mwea_TMS_model_array_annual',
                                                                   file_name='mean_FA_0020_raw_mwea',period='2000_2020')
    mean_FA_1020_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                   key_value='FA_mwea_TMS_model_array_annual',
                                                                   file_name='mean_FA_1020_raw_mwea',period='2010_2020')
    mean_FA_0010_raw_mwea = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                   key_value='FA_mwea_TMS_model_array_annual',
                                                                   file_name='mean_FA_0010_raw_mwea',period='2000_2010')
    
    # == Step 3: extract the dataframes for all keys in the all_glac_data_stats ====
    result_dataframes = stats_t.extract_dataframes(all_glac_data_stats)
    # Define a list of tuples containing DataFrame names and their corresponding output file names
    stats_to_extract = [
        # Mean dL/dt statistics
        ('mean_stats_20002010_lengthchange_dLdt_model_array_annual_myr', 'mean_0010_dLdt_SQ_myr'),
        ('mean_stats_20102020_lengthchange_dLdt_model_array_annual_myr', 'mean_1020_dLdt_SQ_myr'),
        ('mean_stats_20002020_lengthchange_dLdt_model_array_annual_myr', 'mean_0020_dLdt_SQ_myr'),

        # Sum of dL/dt statistics
        ('sum_stats_20002010_lengthchange_dLdt_model_array_annual_myr', 'sum_0010_dL_SQ_m'),
        ('sum_stats_20102020_lengthchange_dLdt_model_array_annual_myr', 'sum_1020_dL_SQ_m'),
        ('sum_stats_20002020_lengthchange_dLdt_model_array_annual_myr', 'sum_0020_dL_SQ_m'),

        # Mean TMS length change statistics
        ('mean_stats_20002010_lengthchange_m_TMS_model_array_annual', 'mean_0010_dLdt_FL_myr'),
        ('mean_stats_20102020_lengthchange_m_TMS_model_array_annual', 'mean_1020_dLdt_FL_myr'),
        ('mean_stats_20002020_lengthchange_m_TMS_model_array_annual', 'mean_0020_dLdt_FL_myr'),

        # Sum of TMS length change statistics
        ('sum_stats_20002010_lengthchange_m_TMS_model_array_annual', 'sum_0010_dL_FL_m'),
        ('sum_stats_20102020_lengthchange_m_TMS_model_array_annual', 'sum_1020_dL_FL_m'),
        ('sum_stats_20002020_lengthchange_m_TMS_model_array_annual', 'sum_0020_dL_FL_m'),

        # Mean calving flux statistics
        ('mean_stats_20002010_calving_flux_Gta_TMS_model_array_annual', 'mean_0010_FA_Gta'),
        ('mean_stats_20102020_calving_flux_Gta_TMS_model_array_annual', 'mean_1020_FA_Gta'),
        ('mean_stats_20002020_calving_flux_Gta_TMS_model_array_annual', 'mean_0020_FA_Gta'),

        # Sum of calving flux statistics
        ('sum_stats_20002010_calving_flux_Gta_TMS_model_array_annual', 'sum_0010_FA_Gt'),
        ('sum_stats_20102020_calving_flux_Gta_TMS_model_array_annual', 'sum_1020_FA_Gt'),
        ('sum_stats_20002020_calving_flux_Gta_TMS_model_array_annual', 'sum_0020_FA_Gt'),

        # Mean mass balance clim statistics
        ('mean_stats_20002010_massbalclim_TMS_model_array_annual_mwea', 'mean_0010_mb_clim_mwea'),
        ('mean_stats_20102020_massbalclim_TMS_model_array_annual_mwea', 'mean_1020_mb_clim_mwea'),
        ('mean_stats_20002020_massbalclim_TMS_model_array_annual_mwea', 'mean_0020_mb_clim_mwea'),

        # Sum of mass balance clim statistics
        ('sum_stats_20002010_massbalclim_TMS_model_array_annual_mwea', 'sum_0010_mb_clim_mwe'),
        ('sum_stats_20102020_massbalclim_TMS_model_array_annual_mwea', 'sum_1020_mb_clim_mwe'),
        ('sum_stats_20002020_massbalclim_TMS_model_array_annual_mwea', 'sum_0020_mb_clim_mwe'),

        # Mean mass balance total statistics
        ('mean_stats_20002010_massbaltotal_TMS_model_array_annual_mwea', 'mean_0010_mb_total_mwea'),
        ('mean_stats_20102020_massbaltotal_TMS_model_array_annual_mwea', 'mean_1020_mb_total_mwea'),
        ('mean_stats_20002020_massbaltotal_TMS_model_array_annual_mwea', 'mean_0020_mb_total_mwea'),

        # Sum of mass balance total statistics
        ('sum_stats_20002010_massbaltotal_TMS_model_array_annual_mwea', 'sum_0010_mb_total_mwe'),
        ('sum_stats_20102020_massbaltotal_TMS_model_array_annual_mwea', 'sum_1020_mb_total_mwe'),
        ('sum_stats_20002020_massbaltotal_TMS_model_array_annual_mwea', 'sum_0020_mb_total_mwe'),

        # Mean mass balance clim GTA statistics
        ('mean_stats_20002010_massbalclim_TMS_model_array_annual_gta', 'mean_0010_mb_clim_Gta'),
        ('mean_stats_20102020_massbalclim_TMS_model_array_annual_gta', 'mean_1020_mb_clim_Gta'),
        ('mean_stats_20002020_massbalclim_TMS_model_array_annual_gta', 'mean_0020_mb_clim_Gta'),
        # Sum of mass balance clim GTA statistics
        ('sum_stats_20002010_massbalclim_TMS_model_array_annual_gta', 'sum_0010_mb_clim_Gt'),
        ('sum_stats_20102020_massbalclim_TMS_model_array_annual_gta', 'sum_1020_mb_clim_Gt'),
        ('sum_stats_20002020_massbalclim_TMS_model_array_annual_gta', 'sum_0020_mb_clim_Gt'),

        # Mean mass balance total GTA statistics
        ('mean_stats_20002010_massbaltotal_TMS_model_array_annual_gta', 'mean_0010_mb_total_Gta'),
        ('mean_stats_20102020_massbaltotal_TMS_model_array_annual_gta', 'mean_1020_mb_total_Gta'),
        ('mean_stats_20002020_massbaltotal_TMS_model_array_annual_gta', 'mean_0020_mb_total_Gta'),
        # Sum of mass balance total GTA statistics
        ('sum_stats_20002010_massbaltotal_TMS_model_array_annual_gta', 'sum_0010_mb_total_Gt'),
        ('sum_stats_20102020_massbaltotal_TMS_model_array_annual_gta', 'sum_1020_mb_total_Gt'),
        ('sum_stats_20002020_massbaltotal_TMS_model_array_annual_gta', 'sum_0020_mb_total_Gt'),

        # Mean FA statistics
        ('mean_stats_20002010_FA_mwea_TMS_model_array_annual', 'mean_0010_FA_mwea'),
        ('mean_stats_20102020_FA_mwea_TMS_model_array_annual', 'mean_1020_FA_mwea'),
        ('mean_stats_20002020_FA_mwea_TMS_model_array_annual', 'mean_0020_FA_mwea'),

        # Sum of FA statistics
        ('sum_stats_20002010_FA_mwea_TMS_model_array_annual', 'sum_0010_FA_mwe'),
        ('sum_stats_20102020_FA_mwea_TMS_model_array_annual', 'sum_1020_FA_mwe'),
        ('sum_stats_20002020_FA_mwea_TMS_model_array_annual', 'sum_0020_FA_mwe'),

        # Mean velocity at calving front statistics
        ('mean_stats_20002010_velocity_at_calvingfront_model_array_annual_myr', 'mean_0010_vel_myr'),
        ('mean_stats_20102020_velocity_at_calvingfront_model_array_annual_myr', 'mean_1020_vel_myr'),
        ('mean_stats_20002020_velocity_at_calvingfront_model_array_annual_myr', 'mean_0020_vel_myr'),

        # Mean thickness at calving front statistics
        ('mean_stats_20002010_thickness_at_calvingfront_model_array_annual_m', 'mean_0010_thickness_m'),
        ('mean_stats_20102020_thickness_at_calvingfront_model_array_annual_m', 'mean_1020_thickness_m'),
        ('mean_stats_20002020_thickness_at_calvingfront_model_array_annual_m', 'mean_0020_thickness_m'),

        # Mean width at calving front statistics
        ('mean_stats_20002010_width_at_calvingfront_model_array_annual_m', 'mean_0010_width_m'),
        ('mean_stats_20102020_width_at_calvingfront_model_array_annual_m', 'mean_1020_width_m'),
        ('mean_stats_20002020_width_at_calvingfront_model_array_annual_m', 'mean_0020_width_m'),

        # Mean volume at BSL statistics
        ('mean_stats_20002010_volume_bsl_model_array_annual_m3', 'mean_0010_vol_bsl_m3'),
        ('mean_stats_20102020_volume_bsl_model_array_annual_m3', 'mean_1020_vol_bsl_m3'),
        ('mean_stats_20002020_volume_bsl_model_array_annual_m3', 'mean_0020_vol_bsl_m3'),

        # Sum of volume at BSL statistics
        ('sum_stats_20002010_volume_bsl_model_array_annual_m3', 'sum_0010_vol_bsl_m3'),
        ('sum_stats_20102020_volume_bsl_model_array_annual_m3', 'sum_1020_vol_bsl_m3'),
        ('sum_stats_20002020_volume_bsl_model_array_annual_m3', 'sum_0020_vol_bsl_m3'),

        # Mean volume at BWL statistics
        ('mean_stats_20002010_volume_bwl_model_array_annual_m3', 'mean_0010_vol_bwl_m3'),
        ('mean_stats_20102020_volume_bwl_model_array_annual_m3', 'mean_1020_vol_bwl_m3'),
        ('mean_stats_20002020_volume_bwl_model_array_annual_m3', 'mean_0020_vol_bwl_m3'),

        # Sum of volume at BWL statistics
        ('sum_stats_20002010_volume_bwl_model_array_annual_m3', 'sum_0010_vol_bwl_m3'),
        ('sum_stats_20102020_volume_bwl_model_array_annual_m3', 'sum_1020_vol_bwl_m3'),
        ('sum_stats_20002020_volume_bwl_model_array_annual_m3', 'sum_0020_vol_bwl_m3')
    ]

    # Extract and save data in a loop
    # Dictionary to hold the extracted DataFrames
    Model_output_stats_dfs = {}
    for df_name, file_name in stats_to_extract:
        Model_output_stats_dfs[file_name]= stats_t.extract_and_save_df(dataframes_dict=result_dataframes,
                            df_name=df_name,
                            output_path=postpro_output_fp_region,
                            file_name=file_name,
                            file_format='csv')
    # extrac the total length change over the period 2000-2010/2010-2020/2000-2020,
    sum_0010_dL_SQ_m =Model_output_stats_dfs['sum_0010_dL_SQ_m']
    sum_1020_dL_SQ_m =Model_output_stats_dfs['sum_1020_dL_SQ_m']
    sum_0020_dL_SQ_m =Model_output_stats_dfs['sum_0020_dL_SQ_m']
    # extract the mean mb clim over the period 2000-2010/2010-2020/2000-2020,
    mean_0010_mb_clim_mwea = Model_output_stats_dfs['mean_0010_mb_clim_mwea']
    mean_1020_mb_clim_mwea = Model_output_stats_dfs['mean_1020_mb_clim_mwea']
    mean_0020_mb_clim_mwea = Model_output_stats_dfs['mean_0020_mb_clim_mwea']
    # extract the mean mb total over the period 2000-2010/2010-2020/2000-2020,
    mean_0010_mb_total_mwea = Model_output_stats_dfs['mean_0010_mb_total_mwea']
    mean_1020_mb_total_mwea = Model_output_stats_dfs['mean_1020_mb_total_mwea']
    mean_0020_mb_total_mwea = Model_output_stats_dfs['mean_0020_mb_total_mwea']
    # extract the mean mb clim GTA over the period 2000-2010/2010-2020/2000-2020,
    mean_0010_mb_clim_Gta = Model_output_stats_dfs['mean_0010_mb_clim_Gta']
    mean_1020_mb_clim_Gta = Model_output_stats_dfs['mean_1020_mb_clim_Gta']
    mean_0020_mb_clim_Gta = Model_output_stats_dfs['mean_0020_mb_clim_Gta']
    # extract the mean mb total GTA over the period 2000-2010/2010-2020/2000-2020,
    mean_0010_mb_total_Gta = Model_output_stats_dfs['mean_0010_mb_total_Gta']
    mean_1020_mb_total_Gta = Model_output_stats_dfs['mean_1020_mb_total_Gta']
    mean_0020_mb_total_Gta = Model_output_stats_dfs['mean_0020_mb_total_Gta']
    # extract the mean FA over the period 2000-2010/2010-2020/2000-2020,
    mean_0010_FA_Gta = Model_output_stats_dfs['mean_0010_FA_Gta']
    mean_1020_FA_Gta = Model_output_stats_dfs['mean_1020_FA_Gta']
    mean_0020_FA_Gta = Model_output_stats_dfs['mean_0020_FA_Gta']
    # extract the mean FA mwea over the period 2000-2010/2010-2020/2000-2020,
    mean_0010_FA_mwea = Model_output_stats_dfs['mean_0010_FA_mwea']
    mean_1020_FA_mwea = Model_output_stats_dfs['mean_1020_FA_mwea']
    mean_0020_FA_mwea = Model_output_stats_dfs['mean_0020_FA_mwea']

    # == Step 4: Load the observation data ====
    # read the observation data from the file, based on the fucntion in the statistic_tool.py
    (fa_obs_20002010_gta, fa_obs_20102020_gta, fa_obs_20002020_gta,
     mb_obs_20002010_gta, mb_obs_20102020_gta, mb_obs_20002020_gta,
     fa_obs_20002010_mwea, fa_obs_20102020_mwea, fa_obs_20002020_mwea,
     mb_obs_20002010_mwea, mb_obs_20102020_mwea, mb_obs_20002020_mwea,
     mb_obs_20002020_mwea_corr, dLdt_20002020) = stats_t.read_load_obs_unc(obs_data_fp)

    # generate the total length change over the period 2000-2010/2010-2020/2000-2020,
    length_change_obs = stats_t.calculate_length_change(dLdt_20002020, interval_years=10)
    # Expand into long format then split
    exploded = length_change_obs.explode(['length_change', 'length_change_unc']).reset_index(drop=True)
    length_change_obs_20002010, length_change_obs_20102020 = exploded.iloc[0::2].reset_index(drop=True), exploded.iloc[1::2].reset_index(drop=True)
    length_change_obs_20002020 =stats_t.calculate_length_change(dLdt_20002020, interval_years=20)
    exploded_0020 = length_change_obs_20002020.explode(['length_change', 'length_change_unc']).reset_index(drop=True)
    length_change_obs_20002020 = exploded_0020.iloc[0::1].reset_index(drop=True)


    # == Step 5: split the rgiid for good and bad AMIS ==
    rgiid_good = stats_t.extract_rgi_ids(filepath=AMIS_statis_fp_region,filename='Good_AMIS.txt')
    if rgiid_good is not None and not rgiid_good.empty:
        rgiid_good = rgiid_good.sort_values('rgiid')
    rgiid_bad = stats_t.extract_rgi_ids(filepath=AMIS_statis_fp_region,filename='Bad_AMIS.txt')
    if rgiid_bad is not None and not rgiid_bad.empty:
        rgiid_bad = rgiid_bad.sort_values('rgiid')

    # == Step 6: Visualization of the calibration output, all AMIS together ==
    # MB_clim_20102020
    Vis_ts.plot_cdf_and_one_to_one(observed_df = mb_obs_20102020_mwea, modeled_df = mean_1020_mb_clim_mwea, modeled_df_raw= mean_MB_clim_1020_raw_mwea,
                            obs_name = 'mb_clim_mwea' , obs_unc_name = 'mb_clim_mwea_err', modeled_key = 'massbalclim_TMS_model_array_annual_mwea', item_name = 'Climatic mass balance (m w.e. a$^{-1}$)',
                            period = '2010-2020',Xlabel = 'Climatic mass balance (m w.e. a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (m w.e. a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,legend_index = False)
    # MB_clim_20002010
    Vis_ts.plot_cdf_and_one_to_one(observed_df = mb_obs_20002010_mwea, modeled_df = mean_0010_mb_clim_mwea, modeled_df_raw= mean_MB_clim_0010_raw_mwea,
                            obs_name = 'mb_clim_mwea' , obs_unc_name = 'mb_clim_mwea_err', modeled_key = 'massbalclim_TMS_model_array_annual_mwea', item_name = 'Climatic mass balance (m w.e. a$^{-1}$)',
                            period = '2000-2010',Xlabel = 'Climatic mass balance (m w.e. a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (m w.e. a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4), 
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,legend_index = False)
    # MB_clim_20002020
    Vis_ts.plot_cdf_and_one_to_one(observed_df = mb_obs_20002020_mwea, modeled_df = mean_0020_mb_clim_mwea, modeled_df_raw= mean_MB_clim_0020_raw_mwea,
                            obs_name = 'mb_clim_mwea' , obs_unc_name = 'mb_clim_mwea_err',
                            modeled_key = 'massbalclim_TMS_model_array_annual_mwea', item_name = 'Climatic mass balance (m w.e. a$^{-1}$)',
                            period = '2000-2020',Xlabel = 'Climatic mass balance (m w.e. a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (m w.e. a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,legend_index = False)
    # dL 20002010
    Vis_ts.plot_cdf_and_one_to_one(observed_df = length_change_obs_20002010, modeled_df = sum_0010_dL_SQ_m, modeled_df_raw= sum_dL_SQ_0010_raw_m,
                                obs_name = 'length_change' , obs_unc_name = 'length_change_unc', modeled_key = 'lengthchange_dLdt_model_array_annual_myr', item_name = 'Length change (km)',
                                period = '2000-2010',Xlabel = 'Length change (km, observed)',Ylabel = 'Length change (km, modeled)', 
                            Xlim = (-5,3), Ylim = (-5,3),subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path = postpro_output_fp_region, save_name=None)
    # dL 20102020
    Vis_ts.plot_cdf_and_one_to_one(observed_df = length_change_obs_20102020, modeled_df = sum_1020_dL_SQ_m, modeled_df_raw= sum_dL_SQ_1020_raw_m,
                                obs_name = 'length_change' , obs_unc_name = 'length_change_unc', modeled_key = 'lengthchange_dLdt_model_array_annual_myr', item_name = 'Length change (km)',
                                period = '2010-2020',Xlabel = 'Length change (km, observed)',Ylabel = 'Length change (km, modeled)', 
                            Xlim = (-5,3), Ylim = (-5,3),subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path = postpro_output_fp_region, save_name=None)
    # dL 20002020
    Vis_ts.plot_cdf_and_one_to_one(observed_df = length_change_obs_20002020, modeled_df = sum_0020_dL_SQ_m, modeled_df_raw= sum_dL_SQ_0020_raw_m,
                                obs_name = 'length_change' , obs_unc_name = 'length_change_unc', modeled_key = 'lengthchange_dLdt_model_array_annual_myr', item_name = 'Length change (km)',
                                period = '2000-2020',Xlabel = 'Length change (km, observed)',Ylabel = 'Length change (km, modeled)', 
                            Xlim = (-5.5,4), Ylim = (-5.5,4),subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path = postpro_output_fp_region, save_name=None)
    
    # FA 20002010
    Vis_ts.plot_cdf_and_one_to_one(observed_df = fa_obs_20002010_gta, modeled_df = mean_0010_FA_Gta, modeled_df_raw= mean_FA_0010_raw_Gta,
                            obs_name = 'fa_gta_obs' , obs_unc_name = 'fa_gta_obs_unc', modeled_key = 'calving_flux_Gta_TMS_model_array_annual', item_name = 'Frontal ablation (Gt a$^{-1}$)',
                            period = '2000-2010',Xlabel = 'Frontal ablation (Gt a$^{-1}$, observed)',Ylabel = 'Frontal ablation (Gt a$^{-1}$, modeled)', 
                        Xlim = (0,0.8), Ylim = (0,0.8),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,legend_index = False)
    # FA 20102020
    Vis_ts.plot_cdf_and_one_to_one(observed_df = fa_obs_20102020_gta, modeled_df = mean_1020_FA_Gta, modeled_df_raw= mean_FA_1020_raw_Gta,
                            obs_name = 'fa_gta_obs' , obs_unc_name = 'fa_gta_obs_unc', modeled_key = 'calving_flux_Gta_TMS_model_array_annual', item_name = 'Frontal ablation (Gt a$^{-1}$)',
                            period = '2010-2020',Xlabel = 'Frontal ablation (Gt a$^{-1}$, observed)',Ylabel = 'Frontal ablation (Gt a$^{-1}$, modeled)', 
                        Xlim = (0,0.8), Ylim = (0,0.8),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,legend_index = False)
    # FA 20002020
    Vis_ts.plot_cdf_and_one_to_one(observed_df = fa_obs_20002020_gta, modeled_df = mean_0020_FA_Gta, modeled_df_raw= mean_FA_0020_raw_Gta,
                            obs_name = 'fa_gta_obs' , obs_unc_name = 'fa_gta_obs_unc', modeled_key = 'calving_flux_Gta_TMS_model_array_annual', item_name = 'Frontal ablation (Gt a$^{-1}$)',
                            period = '2000-2020',Xlabel = 'Frontal ablation (Gt a$^{-1}$, observed)',Ylabel = 'Frontal ablation (Gt a$^{-1}$, modeled)', 
                        Xlim = (0,0.8), Ylim = (0,0.8),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,legend_index = False)
    

    # == Step 7: Visualization of the calibration output, all Good_Bad seperately with different colors ==
    # MB_clim_20102020
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20102020_mwea, modeled_df = mean_1020_mb_clim_mwea, modeled_df_raw= mean_MB_clim_1020_raw_mwea,
                            obs_name = 'mb_clim_mwea' , obs_unc_name = 'mb_clim_mwea_err', modeled_key = 'massbalclim_TMS_model_array_annual_mwea', item_name = 'Climatic mass balance (m w.e. a$^{-1}$)',
                            period = '2010-2020',reg_id = reg_id,Xlabel = 'Climatic mass balance (m w.e. a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (m w.e. a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20002010
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20002010_mwea, modeled_df = mean_0010_mb_clim_mwea, modeled_df_raw= mean_MB_clim_0010_raw_mwea,
                            obs_name = 'mb_clim_mwea' , obs_unc_name = 'mb_clim_mwea_err', modeled_key = 'massbalclim_TMS_model_array_annual_mwea', item_name = 'Climatic mass balance (m w.e. a$^{-1}$)',
                            period = '2000-2010',reg_id = reg_id,Xlabel = 'Climatic mass balance (m w.e. a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (m w.e. a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20002020
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20002020_mwea, modeled_df = mean_0020_mb_clim_mwea, modeled_df_raw= mean_MB_clim_0020_raw_mwea,
                            obs_name = 'mb_clim_mwea' , obs_unc_name = 'mb_clim_mwea_err',
                            modeled_key = 'massbalclim_TMS_model_array_annual_mwea', item_name = 'Climatic mass balance (m w.e. a$^{-1}$)',
                            period = '2000-2020',reg_id = reg_id,Xlabel = 'Climatic mass balance (m w.e. a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (m w.e. a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20102020 gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20102020_gta, modeled_df = mean_1020_mb_clim_Gta, modeled_df_raw= mean_MB_clim_1020_raw_gta,
                            obs_name = 'mb_clim_gta' , obs_unc_name = 'mb_clim_gta_err', modeled_key = 'massbalclim_TMS_model_array_annual_gta', item_name = 'Climatic mass balance (Gt a$^{-1}$)',
                            period = '2010-2020',reg_id = reg_id,Xlabel = 'Climatic mass balance (Gt a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (Gt a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20002010 gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20002010_gta, modeled_df = mean_0010_mb_clim_Gta, modeled_df_raw= mean_MB_clim_0010_raw_gta,
                            obs_name = 'mb_clim_gta' , obs_unc_name = 'mb_clim_gta_err', modeled_key = 'massbalclim_TMS_model_array_annual_gta', item_name = 'Climatic mass balance (Gt a$^{-1}$)',
                            period = '2000-2010',reg_id = reg_id,Xlabel = 'Climatic mass balance (Gt a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (Gt a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20002020 gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20002020_gta, modeled_df = mean_0020_mb_clim_Gta, modeled_df_raw= mean_MB_clim_0020_raw_gta,
                            obs_name = 'mb_clim_gta' , obs_unc_name = 'mb_clim_gta_err', modeled_key = 'massbalclim_TMS_model_array_annual_gta', item_name = 'Climatic mass balance (Gt a$^{-1}$)',
                            period = '2000-2020',reg_id = reg_id,Xlabel = 'Climatic mass balance (Gt a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (Gt a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # dL 20002010
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = length_change_obs_20002010, modeled_df = sum_0010_dL_SQ_m, modeled_df_raw= sum_dL_SQ_0010_raw_m,
                            obs_name = 'length_change' , obs_unc_name = 'length_change_unc', modeled_key = 'lengthchange_dLdt_model_array_annual_myr', item_name = 'Length change (km)',
                            period = '2000-2010',reg_id = reg_id,Xlabel = 'Length change (km, observed)',Ylabel = 'Length change (km, modeled)', 
                        Xlim = (-5,3), Ylim = (-5,3),subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = True,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # dL 20102020
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = length_change_obs_20102020, modeled_df = sum_1020_dL_SQ_m, modeled_df_raw= sum_dL_SQ_1020_raw_m,
                            obs_name = 'length_change' , obs_unc_name = 'length_change_unc', modeled_key = 'lengthchange_dLdt_model_array_annual_myr', item_name = 'Length change (km)',
                            period = '2010-2020',reg_id = reg_id,Xlabel = 'Length change (km, observed)',Ylabel = 'Length change (km, modeled)', 
                        Xlim = (-5,3), Ylim = (-5,3),subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = True,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # dL 20002020
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = length_change_obs_20002020, modeled_df = sum_0020_dL_SQ_m, modeled_df_raw= sum_dL_SQ_0020_raw_m,
                            obs_name = 'length_change' , obs_unc_name = 'length_change_unc', modeled_key = 'lengthchange_dLdt_model_array_annual_myr', item_name = 'Length change (km)',
                            period = '2000-2020',reg_id = reg_id,Xlabel = 'Length change (km, observed)',Ylabel = 'Length change (km, modeled)', 
                        Xlim = (-5.5,4), Ylim = (-5.5,4),subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = True,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # FA 20002010_mwea
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = fa_obs_20002010_mwea, modeled_df = mean_0010_FA_mwea, modeled_df_raw= mean_FA_0010_raw_mwea,
                            obs_name = 'fa_mwea_obs' , obs_unc_name = 'fa_mwea_obs_unc', modeled_key = 'FA_mwea_TMS_model_array_annual', item_name = 'Frontal ablation (m w.e. a$^{-1}$)',
                            period = '2000-2010',reg_id = reg_id,Xlabel = 'Frontal ablation (m w.e. a$^{-1}$, observed)',Ylabel = 'Frontal ablation (m w.e. a$^{-1}$, modeled)', 
                        Xlim = (0,6), Ylim = (0,6),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = True,Good_bad = True,
                            zoom_xlim = (0,1), zoom_ylim = (0,1),zoom_position = [0.7, 0.7, 0.28, 0.28],zoom_ticklabels = True)
    # FA 20102020_mwea
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = fa_obs_20102020_mwea, modeled_df = mean_1020_FA_mwea, modeled_df_raw= mean_FA_1020_raw_mwea,
                            obs_name = 'fa_mwea_obs' , obs_unc_name = 'fa_mwea_obs_unc', modeled_key = 'FA_mwea_TMS_model_array_annual', item_name = 'Frontal ablation (m w.e. a$^{-1}$)',
                            period = '2010-2020',reg_id = reg_id,Xlabel = 'Frontal ablation (m w.e. a$^{-1}$, observed)',Ylabel = 'Frontal ablation (m w.e. a$^{-1}$, modeled)', 
                        Xlim = (0,6), Ylim = (0,6),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = True,Good_bad = True,
                            zoom_xlim = (0,1), zoom_ylim = (0,1),zoom_position = [0.7, 0.7, 0.28, 0.28],zoom_ticklabels = True)
    # FA 20002020_mwea
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = fa_obs_20002020_mwea, modeled_df = mean_0020_FA_mwea, modeled_df_raw= mean_FA_0020_raw_mwea,
                            obs_name = 'fa_mwea_obs' , obs_unc_name = 'fa_mwea_obs_unc', modeled_key = 'FA_mwea_TMS_model_array_annual', item_name = 'Frontal ablation (m w.e. a$^{-1}$)',
                            period = '2000-2020',reg_id = reg_id,Xlabel = 'Frontal ablation (m w.e. a$^{-1}$, observed)',Ylabel = 'Frontal ablation (m w.e. a$^{-1}$, modeled)', 
                        Xlim = (0,6), Ylim = (0,6),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = True,Good_bad = True,
                            zoom_xlim = (0,1), zoom_ylim = (0,1),zoom_position = [0.7, 0.7, 0.28, 0.28],zoom_ticklabels = True)
    # FA 20002010_Gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = fa_obs_20002010_gta, modeled_df = mean_0010_FA_Gta, modeled_df_raw= mean_FA_0010_raw_Gta,
                            obs_name = 'fa_gta_obs' , obs_unc_name = 'fa_gta_obs_unc', modeled_key = 'calving_flux_Gta_TMS_model_array_annual', item_name = 'Frontal ablation (Gt a$^{-1}$)',
                            period = '2000-2010',reg_id = reg_id,Xlabel = 'Frontal ablation (Gt a$^{-1}$, observed)',Ylabel = 'Frontal ablation (Gt a$^{-1}$, modeled)', 
                        Xlim = (0,0.8), Ylim = (0,0.8),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = True,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # FA 20102020_Gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = fa_obs_20102020_gta, modeled_df = mean_1020_FA_Gta, modeled_df_raw= mean_FA_1020_raw_Gta,
                            obs_name = 'fa_gta_obs' , obs_unc_name = 'fa_gta_obs_unc', modeled_key = 'calving_flux_Gta_TMS_model_array_annual', item_name = 'Frontal ablation (Gt a$^{-1}$)',
                            period = '2010-2020',reg_id = reg_id,Xlabel = 'Frontal ablation (Gt a$^{-1}$, observed)',Ylabel = 'Frontal ablation (Gt a$^{-1}$, modeled)', 
                        Xlim = (0,0.8), Ylim = (0,0.8),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = True,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # FA 20002020_Gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = fa_obs_20002020_gta, modeled_df = mean_0020_FA_Gta, modeled_df_raw= mean_FA_0020_raw_Gta,
                            obs_name = 'fa_gta_obs' , obs_unc_name = 'fa_gta_obs_unc', modeled_key = 'calving_flux_Gta_TMS_model_array_annual', item_name = 'Frontal ablation (Gt a$^{-1}$)',
                            period = '2000-2020',reg_id = reg_id,Xlabel = 'Frontal ablation (Gt a$^{-1}$, observed)',Ylabel = 'Frontal ablation (Gt a$^{-1}$, modeled)', 
                        Xlim = (0,0.8), Ylim = (0,0.8),subplot_label_L ='e', subplot_label_R = 'f',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = True,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)


    # == Step 8: load the obs and prior and posterior weighted model output,calculate RMSE/DELTA RMSE of modeled and obs between posterior and prior  ==
    # Initialize the obs dLdt 
    Obs_dLdt = {'rgiid': [],'dLdt_myr_obs': [],'dLdt_myr_unc_obs': []}
    # Iterate through the DataFrame rows
    for _, row in dLdt_20002020.iterrows():
        # Process RGIId
        rgiid_parts = row['RGIId'].split('-')[1].split('.')
        rgiid = f"{int(rgiid_parts[0])}.{rgiid_parts[1]}"
        try:
            # Safely convert string lists to numpy arrays
            dLdt = np.array(ast.literal_eval(row['dLdt_m_per_yr']), dtype=np.float64)
            dLdt_unc = np.array(ast.literal_eval(row['dLdt_m_per_yr_unc']), dtype=np.float64)
        except (ValueError, SyntaxError) as e:
            print(f"Error processing row {row['RGIId']}: {e}")
            continue  # Skip to the next iteration if conversion fails

        # Fill the observation lists
        Obs_dLdt['rgiid'].append(rgiid)
        Obs_dLdt['dLdt_myr_obs'].append(dLdt)
        Obs_dLdt['dLdt_myr_unc_obs'].append(dLdt_unc)
    Obs_dLdt_df = pd.DataFrame(Obs_dLdt)

    # read the prior weighted model output
    Prior_weighted = stats_t.read_prior_compute_weighted_save_region(model_output_region_fp=model_output_fp_region, AMIS_fp = model_AMIS_fp_region, data_index='Annual',
                                            file_name=None, file_format='json')
    # print the keys in Prior_weighted
    if log_level == 'DEBUG':
        print("Keys in Prior_weighted:", Prior_weighted.keys())
    Prior_weighted_df = pd.DataFrame(Prior_weighted)
    Prior_weighted_df = Prior_weighted_df.sort_values(by='rgiid').reset_index(drop=True)
    # dLdt annual weighted prior
    dLdt_annual_weighted_Prior = pd.DataFrame()
    dLdt_annual_weighted_Prior ['rgiid'] =Prior_weighted_df['rgiid']
    dLdt_annual_weighted_Prior ['dLdt_annual_myr_prior'] =Prior_weighted_df['lengthchange_dLdt_model_array_annual_myr_weighted']
    # MB_Clim_mwea prior
    MB_clim_annual_weighted_Prior = pd.DataFrame()
    MB_clim_annual_weighted_Prior['rgiid'] = Prior_weighted_df['rgiid']
    MB_clim_annual_weighted_Prior['mb_annul_mwea_prior'] = Prior_weighted_df['massbalclim_TMS_model_array_annual_mwea_weighted']
    # FA_gta prior
    FA_annual_weighted_Prior = pd.DataFrame()
    FA_annual_weighted_Prior['rgiid'] = Prior_weighted_df['rgiid']
    FA_annual_weighted_Prior['fa_annual_gta_prior'] = Prior_weighted_df['calving_flux_Gta_TMS_model_array_annual_weighted']

    # read the posterior weighted model output
    Poster_weighted = stats_t.load_posterior_weighted_region(model_output_fp_region=model_output_fp_region)
    Poster_weighted_df =pd.DataFrame(Poster_weighted)
    Poster_weighted_df = Poster_weighted_df.sort_values(by='rgiid').reset_index(drop=True)
    if log_level == 'DEBUG':
        print("Keys in Poster_weighted:", Poster_weighted_df.keys())
    # dLdt annual weighted posterior
    dLdt_annual_weighted_Poster = pd.DataFrame()
    dLdt_annual_weighted_Poster ['rgiid'] =Poster_weighted_df['rgiid']
    dLdt_annual_weighted_Poster ['dLdt_annual_myr_poster'] =Poster_weighted_df['lengthchange_dLdt_model_annual_weighted_myr']
    # MB_Clim_mwea prior
    MB_clim_annual_weighted_Poster = pd.DataFrame()
    MB_clim_annual_weighted_Poster['rgiid'] = Poster_weighted_df['rgiid']
    MB_clim_annual_weighted_Poster['mb_annul_mwea_poster'] = Poster_weighted_df['massbalclim_TMS_model_annual_weighted_mwea']
    # FA_gta prior
    FA_annual_weighted_Poster = pd.DataFrame()
    FA_annual_weighted_Poster['rgiid'] = Poster_weighted_df['rgiid']
    FA_annual_weighted_Poster['fa_annual_gta_poster'] = Poster_weighted_df['calving_flux_Gta_TMS_model_annual_weighted']

    # Merge the DataFrames
    # dLdt annual weighted prior and posterior and obs
    dLdt_annual_weighted_poster_prior = pd.merge(dLdt_annual_weighted_Prior, dLdt_annual_weighted_Poster, on='rgiid', how='inner')
    dLdt_annual_weighted_obs_poster_prior = pd.merge(dLdt_annual_weighted_poster_prior, Obs_dLdt_df, on='rgiid', how='inner')

    # MB_clim annual weighted prior and posterior and obs
    # first need to get the 10-years average of the annual data for posterior and prior
    # Then merge the DataFrames
    # Prior annual weighted prior and posterior and obs
    MB_clim_annual_weighted_Prior_20002010 = pd.DataFrame()
    MB_clim_annual_weighted_Prior_20002010['rgiid'] = MB_clim_annual_weighted_Prior['rgiid']
    MB_clim_annual_weighted_Prior_20002010['mb_annul_mwea_prior'] = MB_clim_annual_weighted_Prior['mb_annul_mwea_prior'].apply(lambda x: sum(x[:10]) / 10)   
    MB_clim_annual_weighted_Prior_20102020 = pd.DataFrame()
    MB_clim_annual_weighted_Prior_20102020['rgiid'] = MB_clim_annual_weighted_Prior['rgiid']
    MB_clim_annual_weighted_Prior_20102020['mb_annul_mwea_prior'] = MB_clim_annual_weighted_Prior['mb_annul_mwea_prior'].apply(lambda x: sum(x[10:20]) / 10)
    MB_clim_annual_weighted_Prior_20002020 = pd.DataFrame()
    MB_clim_annual_weighted_Prior_20002020['rgiid'] = MB_clim_annual_weighted_Prior['rgiid']    
    MB_clim_annual_weighted_Prior_20002020['mb_annul_mwea_prior'] = MB_clim_annual_weighted_Prior['mb_annul_mwea_prior'].apply(lambda x: sum(x[:20]) / 20)
    # Posterior annual weighted mb_clim
    MB_clim_annual_weighted_Poster_20002010 = pd.DataFrame()
    MB_clim_annual_weighted_Poster_20002010['rgiid'] = MB_clim_annual_weighted_Poster['rgiid']
    MB_clim_annual_weighted_Poster_20002010['mb_annul_mwea_poster'] = MB_clim_annual_weighted_Poster['mb_annul_mwea_poster'].apply(lambda x: sum(x[:10]) / 10)
    MB_clim_annual_weighted_Poster_20102020 = pd.DataFrame()
    MB_clim_annual_weighted_Poster_20102020['rgiid'] = MB_clim_annual_weighted_Poster['rgiid']
    MB_clim_annual_weighted_Poster_20102020['mb_annul_mwea_poster'] = MB_clim_annual_weighted_Poster['mb_annul_mwea_poster'].apply(lambda x: sum(x[10:20]) / 10)
    MB_clim_annual_weighted_Poster_20002020 = pd.DataFrame()
    MB_clim_annual_weighted_Poster_20002020['rgiid'] = MB_clim_annual_weighted_Poster['rgiid']
    MB_clim_annual_weighted_Poster_20002020['mb_annul_mwea_poster'] = MB_clim_annual_weighted_Poster['mb_annul_mwea_poster'].apply(lambda x: sum(x[:20]) / 20)
    # Merege the 10-years average DataFrames weighted prior and posterior and obs
    MB_clim_annual_weighted_poster_prior_20002010 = pd.merge(MB_clim_annual_weighted_Prior_20002010, MB_clim_annual_weighted_Poster_20002010, on='rgiid', how='inner')
    MB_clim_annual_weighted_poster_prior_20102020 = pd.merge(MB_clim_annual_weighted_Prior_20102020, MB_clim_annual_weighted_Poster_20102020, on='rgiid', how='inner')
    MB_clim_annual_weighted_poster_prior_20002020 = pd.merge(MB_clim_annual_weighted_Prior_20002020, MB_clim_annual_weighted_Poster_20002020, on='rgiid', how='inner')
    MB_clim_annual_weighted_obs_poster_prior_20002010 = pd.merge(MB_clim_annual_weighted_poster_prior_20002010, mb_obs_20002010_mwea, on='rgiid', how='inner')
    MB_clim_annual_weighted_obs_poster_prior_20102020 = pd.merge(MB_clim_annual_weighted_poster_prior_20102020, mb_obs_20102020_mwea, on='rgiid', how='inner')
    MB_clim_annual_weighted_obs_poster_prior_20002020 = pd.merge(MB_clim_annual_weighted_poster_prior_20002020, mb_obs_20002020_mwea, on='rgiid', how='inner')
    # FA annual weighted prior and posterior and obs
    FA_annual_weighted_Prior_20002010 = pd.DataFrame()
    FA_annual_weighted_Prior_20002010['rgiid'] = FA_annual_weighted_Prior['rgiid']
    FA_annual_weighted_Prior_20002010['fa_annual_gta_prior'] = FA_annual_weighted_Prior['fa_annual_gta_prior'].apply(lambda x: sum(x[:10]) / 10)
    FA_annual_weighted_Prior_20102020 = pd.DataFrame()
    FA_annual_weighted_Prior_20102020['rgiid'] = FA_annual_weighted_Prior['rgiid']
    FA_annual_weighted_Prior_20102020['fa_annual_gta_prior'] = FA_annual_weighted_Prior['fa_annual_gta_prior'].apply(lambda x: sum(x[10:20]) / 10)
    FA_annual_weighted_Prior_20002020 = pd.DataFrame()
    FA_annual_weighted_Prior_20002020['rgiid'] = FA_annual_weighted_Prior['rgiid']
    FA_annual_weighted_Prior_20002020['fa_annual_gta_prior'] = FA_annual_weighted_Prior['fa_annual_gta_prior'].apply(lambda x: sum(x[:20]) / 20)
    # FA annual weighted posterior and obs
    FA_annual_weighted_Poster_20002010 = pd.DataFrame()
    FA_annual_weighted_Poster_20002010['rgiid'] = FA_annual_weighted_Poster['rgiid']
    FA_annual_weighted_Poster_20002010['fa_annual_gta_poster'] = FA_annual_weighted_Poster['fa_annual_gta_poster'].apply(lambda x: sum(x[:10]) / 10)
    FA_annual_weighted_Poster_20102020 = pd.DataFrame()
    FA_annual_weighted_Poster_20102020['rgiid'] = FA_annual_weighted_Poster['rgiid']
    FA_annual_weighted_Poster_20102020['fa_annual_gta_poster'] = FA_annual_weighted_Poster['fa_annual_gta_poster'].apply(lambda x: sum(x[10:20]) / 10)
    FA_annual_weighted_Poster_20002020 = pd.DataFrame()
    FA_annual_weighted_Poster_20002020['rgiid'] = FA_annual_weighted_Poster['rgiid']
    FA_annual_weighted_Poster_20002020['fa_annual_gta_poster'] = FA_annual_weighted_Poster['fa_annual_gta_poster'].apply(lambda x: sum(x[:20]) / 20)
    # Merege the 10-years average DataFrames weighted prior and posterior and obs
    FA_annual_weighted_poster_prior_20002010 = pd.merge(FA_annual_weighted_Prior_20002010, FA_annual_weighted_Poster_20002010, on='rgiid', how='inner')
    FA_annual_weighted_poster_prior_20102020 = pd.merge(FA_annual_weighted_Prior_20102020, FA_annual_weighted_Poster_20102020, on='rgiid', how='inner')
    FA_annual_weighted_poster_prior_20002020 = pd.merge(FA_annual_weighted_Prior_20002020, FA_annual_weighted_Poster_20002020, on='rgiid', how='inner')
    FA_annual_weighted_obs_poster_prior_20002010 = pd.merge(FA_annual_weighted_poster_prior_20002010, fa_obs_20002010_gta, on='rgiid', how='inner')
    FA_annual_weighted_obs_poster_prior_20102020 = pd.merge(FA_annual_weighted_poster_prior_20102020, fa_obs_20102020_gta, on='rgiid', how='inner')
    FA_annual_weighted_obs_poster_prior_20002020 = pd.merge(FA_annual_weighted_poster_prior_20002020, fa_obs_20002020_gta, on='rgiid', how='inner')

    # Merge the DataFrames based on the good and bad AMIS (converged and unconverged)
    # dLdt annual weighted prior and posterior and obs
    if rgiid_good is not None:
        #==dLdt
        dLdt_annual_merged_good = pd.merge(dLdt_annual_weighted_obs_poster_prior, rgiid_good, on='rgiid', how='inner')
        # Extract the df for dLdt annual weighted prior and posterior and obs
        dLdt_prior_df_good = dLdt_annual_merged_good ['dLdt_annual_myr_prior']
        dLdt_poster_df_good = dLdt_annual_merged_good ['dLdt_annual_myr_poster']
        dLdt_obs_df_good = dLdt_annual_merged_good ['dLdt_myr_obs']
        dLdt_obs_unc_df_good = dLdt_annual_merged_good ['dLdt_myr_unc_obs']
        dLdt_prior_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_good[i]]),
                                                            model_values= np.array([dLdt_prior_df_good[i]]),
                                                            obs_uncertainty=np.array([dLdt_obs_unc_df_good[i]]),
                                                            adjust_uncertainty=True) for i in range(len(dLdt_obs_df_good))])
        dLdt_prior_0010_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_good[i][0:10]]),
                                                                model_values= np.array([dLdt_prior_df_good[i][0:10]]),
                                                                obs_uncertainty=np.array([dLdt_obs_unc_df_good[i][0:10]]),
                                                                adjust_uncertainty=True) for i in range(len(dLdt_obs_df_good))])
        dLdt_prior_1020_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_good[i][10:20]]),
                                                                model_values= np.array([dLdt_prior_df_good[i][10:20]]),
                                                                obs_uncertainty=np.array([dLdt_obs_unc_df_good[i][10:20]]),
                                                                adjust_uncertainty=True) for i in range(len(dLdt_obs_df_good))])
        dLdt_poster_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_good[i]]),
                                                                model_values= np.array([dLdt_poster_df_good[i]]),
                                                                obs_uncertainty=np.array([dLdt_obs_unc_df_good[i]]),
                                                                adjust_uncertainty=True) for i in range(len(dLdt_obs_df_good))])
        dLdt_poster_0010_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_good[i][0:10]]),
                                                                model_values= np.array([dLdt_poster_df_good[i][0:10]]),
                                                                obs_uncertainty=np.array([dLdt_obs_unc_df_good[i][0:10]]),
                                                                adjust_uncertainty=True) for i in range(len(dLdt_obs_df_good))])
        dLdt_poster_1020_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_good[i][10:20]]),
                                                                model_values= np.array([dLdt_poster_df_good[i][10:20]]),
                                                                obs_uncertainty=np.array([dLdt_obs_unc_df_good[i][10:20]]),
                                                                adjust_uncertainty=True) for i in range(len(dLdt_obs_df_good))])
        # calculate the delta RMSE
        dLdt_delta_rmse_good = dLdt_poster_rmse_good - dLdt_prior_rmse_good
        dLdt_delta_0010_rmse_good = dLdt_poster_0010_rmse_good - dLdt_prior_0010_rmse_good
        dLdt_delta_1020_rmse_good = dLdt_poster_1020_rmse_good - dLdt_prior_1020_rmse_good

        # ==MB_clim
        MB_clim_annual_merged_good_20002010 = pd.merge(MB_clim_annual_weighted_obs_poster_prior_20002010, rgiid_good, on='rgiid', how='inner')
        MB_clim_annual_merged_good_20102020 = pd.merge(MB_clim_annual_weighted_obs_poster_prior_20102020, rgiid_good, on='rgiid', how='inner')
        MB_clim_annual_merged_good_20002020 = pd.merge(MB_clim_annual_weighted_obs_poster_prior_20002020, rgiid_good, on='rgiid', how='inner')
        MB_clim_prior_df_good_20002010 = MB_clim_annual_merged_good_20002010 ['mb_annul_mwea_prior']
        MB_clim_prior_df_good_20102020 = MB_clim_annual_merged_good_20102020 ['mb_annul_mwea_prior']
        MB_clim_prior_df_good_20002020 = MB_clim_annual_merged_good_20002020 ['mb_annul_mwea_prior']
        MB_clim_poster_df_good_20002010 = MB_clim_annual_merged_good_20002010 ['mb_annul_mwea_poster']
        MB_clim_poster_df_good_20102020 = MB_clim_annual_merged_good_20102020 ['mb_annul_mwea_poster']
        MB_clim_poster_df_good_20002020 = MB_clim_annual_merged_good_20002020 ['mb_annul_mwea_poster']
        MB_clim_obs_df_good_20002010 = MB_clim_annual_merged_good_20002010 ['mb_clim_mwea']
        MB_clim_obs_unc_df_good_20002010 = MB_clim_annual_merged_good_20002010 ['mb_clim_mwea_err']
        MB_clim_obs_df_good_20102020 = MB_clim_annual_merged_good_20102020 ['mb_clim_mwea']
        MB_clim_obs_unc_df_good_20102020 = MB_clim_annual_merged_good_20102020 ['mb_clim_mwea_err']
        MB_clim_obs_df_good_20002020 = MB_clim_annual_merged_good_20002020 ['mb_clim_mwea']
        MB_clim_obs_unc_df_good_20002020 = MB_clim_annual_merged_good_20002020 ['mb_clim_mwea_err']

        MB_clim_prior_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_good_20002020[i]]),
                                                                model_values= np.array([MB_clim_prior_df_good_20002020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_good_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_good_20002020))])
        MB_clim_prior_0010_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_good_20002010[i]]),
                                                                model_values= np.array([MB_clim_prior_df_good_20002010[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_good_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_good_20002010))])
        MB_clim_prior_1020_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_good_20102020[i]]),
                                                                model_values= np.array([MB_clim_prior_df_good_20102020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_good_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_good_20102020))])
        MB_clim_poster_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_good_20002020[i]]),
                                                                model_values= np.array([MB_clim_poster_df_good_20002020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_good_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_good_20002020))])
        MB_clim_poster_0010_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_good_20002010[i]]),
                                                                model_values= np.array([MB_clim_poster_df_good_20002010[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_good_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_good_20002010))])
        MB_clim_poster_1020_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_good_20102020[i]]),
                                                                model_values= np.array([MB_clim_poster_df_good_20102020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_good_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_good_20102020))])
        # calculate the delta RMSE
        MB_clim_delta_rmse_good = MB_clim_poster_rmse_good - MB_clim_prior_rmse_good
        MB_clim_delta_0010_rmse_good = MB_clim_poster_0010_rmse_good - MB_clim_prior_0010_rmse_good
        MB_clim_delta_1020_rmse_good = MB_clim_poster_1020_rmse_good - MB_clim_prior_1020_rmse_good

        # ==FA
        FA_annual_merged_good_20002010 = pd.merge(FA_annual_weighted_obs_poster_prior_20002010, rgiid_good, on='rgiid', how='inner')
        FA_annual_merged_good_20102020 = pd.merge(FA_annual_weighted_obs_poster_prior_20102020, rgiid_good, on='rgiid', how='inner')
        FA_annual_merged_good_20002020 = pd.merge(FA_annual_weighted_obs_poster_prior_20002020, rgiid_good, on='rgiid', how='inner')
        FA_prior_df_good_20002010 = FA_annual_merged_good_20002010 ['fa_annual_gta_prior']
        FA_prior_df_good_20102020 = FA_annual_merged_good_20102020 ['fa_annual_gta_prior']
        FA_prior_df_good_20002020 = FA_annual_merged_good_20002020 ['fa_annual_gta_prior']
        FA_poster_df_good_20002010 = FA_annual_merged_good_20002010 ['fa_annual_gta_poster']
        FA_poster_df_good_20102020 = FA_annual_merged_good_20102020 ['fa_annual_gta_poster']
        FA_poster_df_good_20002020 = FA_annual_merged_good_20002020 ['fa_annual_gta_poster']
        FA_obs_df_good_20002010 = FA_annual_merged_good_20002010 ['fa_gta_obs']
        FA_obs_unc_df_good_20002010 = FA_annual_merged_good_20002010 ['fa_gta_obs_unc']
        FA_obs_df_good_20002020 = FA_annual_merged_good_20002020 ['fa_gta_obs']
        FA_obs_unc_df_good_20002020 = FA_annual_merged_good_20002020 ['fa_gta_obs_unc']
        FA_obs_df_good_20102020 = FA_annual_merged_good_20102020 ['fa_gta_obs']
        FA_obs_unc_df_good_20102020 = FA_annual_merged_good_20102020 ['fa_gta_obs_unc']
        FA_prior_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_good_20002020[i]]),
                                                                model_values= np.array([FA_prior_df_good_20002020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_good_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_good_20002020))])
        FA_prior_0010_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_good_20002010[i]]),
                                                                model_values= np.array([FA_prior_df_good_20002010[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_good_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_good_20002010))])
        FA_prior_1020_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_good_20102020[i]]),
                                                                model_values= np.array([FA_prior_df_good_20102020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_good_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_good_20102020))])
        FA_poster_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_good_20002020[i]]),
                                                                model_values= np.array([FA_poster_df_good_20002020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_good_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_good_20002020))])
        FA_poster_0010_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_good_20002010[i]]),
                                                                model_values= np.array([FA_poster_df_good_20002010[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_good_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_good_20002010))])
        FA_poster_1020_rmse_good = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_good_20102020[i]]),
                                                                model_values= np.array([FA_poster_df_good_20102020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_good_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_good_20102020))])
        # calculate the delta RMSE
        FA_delta_rmse_good = FA_poster_rmse_good - FA_prior_rmse_good
        FA_delta_0010_rmse_good = FA_poster_0010_rmse_good - FA_prior_0010_rmse_good
        FA_delta_1020_rmse_good = FA_poster_1020_rmse_good - FA_prior_1020_rmse_good
    else:
        dLdt_prior_df_good = None
        dLdt_poster_df_good = None
        dLdt_obs_df_good = None
        dLdt_obs_unc_df_good = None
        dLdt_prior_rmse_good = None
        dLdt_prior_0010_rmse_good = None
        dLdt_prior_1020_rmse_good = None
        dLdt_poster_rmse_good = None
        dLdt_poster_0010_rmse_good = None
        dLdt_poster_1020_rmse_good = None
        dLdt_delta_rmse_good = None
        dLdt_delta_0010_rmse_good = None
        dLdt_delta_1020_rmse_good = None
        MB_clim_prior_rmse_good = None
        MB_clim_prior_0010_rmse_good = None
        MB_clim_prior_1020_rmse_good = None
        MB_clim_poster_rmse_good = None
        MB_clim_poster_0010_rmse_good = None
        MB_clim_poster_1020_rmse_good = None
        MB_clim_delta_rmse_good = None
        MB_clim_delta_0010_rmse_good = None
        MB_clim_delta_1020_rmse_good = None
        FA_prior_rmse_good = None
        FA_prior_0010_rmse_good = None
        FA_prior_1020_rmse_good = None
        FA_poster_rmse_good = None
        FA_poster_0010_rmse_good = None
        FA_poster_1020_rmse_good = None
        FA_delta_rmse_good = None
        FA_delta_0010_rmse_good = None
        FA_delta_1020_rmse_good = None

    if rgiid_bad is not None:
        # ==dLdt
        dLdt_annual_merged_bad = pd.merge(dLdt_annual_weighted_obs_poster_prior, rgiid_bad, on='rgiid', how='inner')
        dLdt_prior_df_bad = dLdt_annual_merged_bad ['dLdt_annual_myr_prior']
        dLdt_poster_df_bad = dLdt_annual_merged_bad ['dLdt_annual_myr_poster']
        dLdt_obs_df_bad = dLdt_annual_merged_bad ['dLdt_myr_obs']
        dLdt_obs_unc_df_bad = dLdt_annual_merged_bad ['dLdt_myr_unc_obs']
        dLdt_prior_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_bad[i]]),
                                                              model_values= np.array([dLdt_prior_df_bad[i]]),
                                                              obs_uncertainty=np.array([dLdt_obs_unc_df_bad[i]]),
                                                              adjust_uncertainty=True) for i in range(len(dLdt_obs_df_bad))])
        dLdt_prior_0010_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_bad[i][0:10]]),
                                                               model_values= np.array([dLdt_prior_df_bad[i][0:10]]),
                                                               obs_uncertainty=np.array([dLdt_obs_unc_df_bad[i][0:10]]),
                                                               adjust_uncertainty=True) for i in range(len(dLdt_obs_df_bad))])
        dLdt_prior_1020_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_bad[i][10:20]]),
                                                              model_values= np.array([dLdt_prior_df_bad[i][10:20]]),
                                                              obs_uncertainty=np.array([dLdt_obs_unc_df_bad[i][10:20]]),
                                                              adjust_uncertainty=True) for i in range(len(dLdt_obs_df_bad))])
        dLdt_poster_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_bad[i]]),
                                                              model_values= np.array([dLdt_poster_df_bad[i]]),
                                                              obs_uncertainty=np.array([dLdt_obs_unc_df_bad[i]]),
                                                              adjust_uncertainty=True) for i in range(len(dLdt_obs_df_bad))])
        dLdt_poster_0010_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_bad[i][0:10]]),
                                                                model_values= np.array([dLdt_poster_df_bad[i][0:10]]),
                                                                obs_uncertainty=np.array([dLdt_obs_unc_df_bad[i][0:10]]),
                                                                adjust_uncertainty=True) for i in range(len(dLdt_obs_df_bad))])
        dLdt_poster_1020_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([dLdt_obs_df_bad[i][10:20]]),
                                                               model_values= np.array([dLdt_poster_df_bad[i][10:20]]),
                                                               obs_uncertainty=np.array([dLdt_obs_unc_df_bad[i][10:20]]),
                                                               adjust_uncertainty=True) for i in range(len(dLdt_obs_df_bad))])
        # calculate the delta RMSE
        dLdt_delta_rmse_bad = dLdt_poster_rmse_bad - dLdt_prior_rmse_bad
        dLdt_delta_0010_rmse_bad = dLdt_poster_0010_rmse_bad - dLdt_prior_0010_rmse_bad
        dLdt_delta_1020_rmse_bad = dLdt_poster_1020_rmse_bad - dLdt_prior_1020_rmse_bad

        # ==MB_clim
        MB_clim_annual_merged_bad_20002010 = pd.merge(MB_clim_annual_weighted_obs_poster_prior_20002010, rgiid_bad, on='rgiid', how='inner')
        MB_clim_annual_merged_bad_20102020 = pd.merge(MB_clim_annual_weighted_obs_poster_prior_20102020, rgiid_bad, on='rgiid', how='inner')
        MB_clim_annual_merged_bad_20002020 = pd.merge(MB_clim_annual_weighted_obs_poster_prior_20002020, rgiid_bad, on='rgiid', how='inner')
        MB_clim_prior_df_bad_20002010 = MB_clim_annual_merged_bad_20002010 ['mb_annul_mwea_prior']
        MB_clim_prior_df_bad_20102020 = MB_clim_annual_merged_bad_20102020 ['mb_annul_mwea_prior']
        MB_clim_prior_df_bad_20002020 = MB_clim_annual_merged_bad_20002020 ['mb_annul_mwea_prior']
        MB_clim_poster_df_bad_20002010 = MB_clim_annual_merged_bad_20002010 ['mb_annul_mwea_poster']
        MB_clim_poster_df_bad_20102020 = MB_clim_annual_merged_bad_20102020 ['mb_annul_mwea_poster']
        MB_clim_poster_df_bad_20002020 = MB_clim_annual_merged_bad_20002020 ['mb_annul_mwea_poster']
        MB_clim_obs_df_bad_20002010 = MB_clim_annual_merged_bad_20002010 ['mb_clim_mwea']
        MB_clim_obs_unc_df_bad_20002010 = MB_clim_annual_merged_bad_20002010 ['mb_clim_mwea_err']
        MB_clim_obs_df_bad_20102020 = MB_clim_annual_merged_bad_20102020 ['mb_clim_mwea']
        MB_clim_obs_unc_df_bad_20102020 = MB_clim_annual_merged_bad_20102020 ['mb_clim_mwea_err']
        MB_clim_obs_df_bad_20002020 = MB_clim_annual_merged_bad_20002020 ['mb_clim_mwea']
        MB_clim_obs_unc_df_bad_20002020 = MB_clim_annual_merged_bad_20002020 ['mb_clim_mwea_err']
        MB_clim_prior_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_bad_20002020[i]]),
                                                                model_values= np.array([MB_clim_prior_df_bad_20002020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_bad_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_bad_20002020))])
        MB_clim_prior_0010_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_bad_20002010[i]]),
                                                                model_values= np.array([MB_clim_prior_df_bad_20002010[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_bad_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_bad_20002010))])
        MB_clim_prior_1020_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_bad_20102020[i]]),
                                                                model_values= np.array([MB_clim_prior_df_bad_20102020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_bad_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_bad_20102020))])
        MB_clim_poster_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_bad_20002020[i]]),
                                                                model_values= np.array([MB_clim_poster_df_bad_20002020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_bad_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_bad_20002020))])
        MB_clim_poster_0010_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_bad_20002010[i]]),
                                                                model_values= np.array([MB_clim_poster_df_bad_20002010[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_bad_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_bad_20002010))])
        MB_clim_poster_1020_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([MB_clim_obs_df_bad_20102020[i]]),
                                                                model_values= np.array([MB_clim_poster_df_bad_20102020[i]]),
                                                                obs_uncertainty=np.array([MB_clim_obs_unc_df_bad_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(MB_clim_obs_df_bad_20102020))])
        # calculate the delta RMSE
        MB_clim_delta_rmse_bad = MB_clim_poster_rmse_bad - MB_clim_prior_rmse_bad
        MB_clim_delta_0010_rmse_bad = MB_clim_poster_0010_rmse_bad - MB_clim_prior_0010_rmse_bad
        MB_clim_delta_1020_rmse_bad = MB_clim_poster_1020_rmse_bad - MB_clim_prior_1020_rmse_bad

        # ==FA
        FA_annual_merged_bad_20002010 = pd.merge(FA_annual_weighted_obs_poster_prior_20002010, rgiid_bad, on='rgiid', how='inner')
        FA_annual_merged_bad_20102020 = pd.merge(FA_annual_weighted_obs_poster_prior_20102020, rgiid_bad, on='rgiid', how='inner')
        FA_annual_merged_bad_20002020 = pd.merge(FA_annual_weighted_obs_poster_prior_20002020, rgiid_bad, on='rgiid', how='inner')
        FA_prior_df_bad_20002010 = FA_annual_merged_bad_20002010 ['fa_annual_gta_prior']
        FA_prior_df_bad_20102020 = FA_annual_merged_bad_20102020 ['fa_annual_gta_prior']
        FA_prior_df_bad_20002020 = FA_annual_merged_bad_20002020 ['fa_annual_gta_prior']
        FA_poster_df_bad_20002010 = FA_annual_merged_bad_20002010 ['fa_annual_gta_poster']
        FA_poster_df_bad_20102020 = FA_annual_merged_bad_20102020 ['fa_annual_gta_poster']
        FA_poster_df_bad_20002020 = FA_annual_merged_bad_20002020 ['fa_annual_gta_poster']
        FA_obs_df_bad_20002010 = FA_annual_merged_bad_20002010 ['fa_gta_obs']
        FA_obs_unc_df_bad_20002010 = FA_annual_merged_bad_20002010 ['fa_gta_obs_unc']
        FA_obs_df_bad_20002020 = FA_annual_merged_bad_20002020 ['fa_gta_obs']
        FA_obs_unc_df_bad_20002020 = FA_annual_merged_bad_20002020 ['fa_gta_obs_unc']
        FA_obs_df_bad_20102020 = FA_annual_merged_bad_20102020 ['fa_gta_obs']
        FA_obs_unc_df_bad_20102020 = FA_annual_merged_bad_20102020 ['fa_gta_obs_unc']
        FA_prior_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_bad_20002020[i]]),
                                                                model_values= np.array([FA_prior_df_bad_20002020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_bad_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_bad_20002020))])
        FA_prior_0010_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_bad_20002010[i]]),
                                                                model_values= np.array([FA_prior_df_bad_20002010[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_bad_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_bad_20002010))])
        FA_prior_1020_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_bad_20102020[i]]),
                                                                model_values= np.array([FA_prior_df_bad_20102020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_bad_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_bad_20102020))])
        FA_poster_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_bad_20002020[i]]),
                                                                model_values= np.array([FA_poster_df_bad_20002020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_bad_20002020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_bad_20002020))])
        FA_poster_0010_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_bad_20002010[i]]),
                                                                model_values= np.array([FA_poster_df_bad_20002010[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_bad_20002010[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_bad_20002010))])
        FA_poster_1020_rmse_bad = np.array([stats_t.calculate_rmse_with_unc(obs_values=np.array([FA_obs_df_bad_20102020[i]]),
                                                                model_values= np.array([FA_poster_df_bad_20102020[i]]),
                                                                obs_uncertainty=np.array([FA_obs_unc_df_bad_20102020[i]]),
                                                                adjust_uncertainty=True) for i in range(len(FA_obs_df_bad_20102020))])
        # calculate the delta RMSE
        FA_delta_rmse_bad = FA_poster_rmse_bad - FA_prior_rmse_bad
        FA_delta_0010_rmse_bad = FA_poster_0010_rmse_bad - FA_prior_0010_rmse_bad
        FA_delta_1020_rmse_bad = FA_poster_1020_rmse_bad - FA_prior_1020_rmse_bad

    else:
        dLdt_prior_df_bad = None
        dLdt_poster_df_bad = None
        dLdt_obs_df_bad = None
        dLdt_obs_unc_df_bad = None
        dLdt_prior_rmse_bad = None
        dLdt_prior_0010_rmse_bad = None
        dLdt_prior_1020_rmse_bad = None
        dLdt_poster_rmse_bad = None
        dLdt_poster_0010_rmse_bad = None
        dLdt_poster_1020_rmse_bad = None
        dLdt_delta_rmse_bad = None
        dLdt_delta_0010_rmse_bad = None
        dLdt_delta_1020_rmse_bad = None
        MB_clim_prior_rmse_bad = None
        MB_clim_prior_0010_rmse_bad = None
        MB_clim_prior_1020_rmse_bad = None
        MB_clim_poster_rmse_bad = None
        MB_clim_poster_0010_rmse_bad = None
        MB_clim_poster_1020_rmse_bad = None
        MB_clim_delta_rmse_bad = None
        MB_clim_delta_0010_rmse_bad = None
        MB_clim_delta_1020_rmse_bad = None
        FA_prior_rmse_bad = None
        FA_prior_0010_rmse_bad = None
        FA_prior_1020_rmse_bad = None
        FA_poster_rmse_bad = None
        FA_poster_0010_rmse_bad = None
        FA_poster_1020_rmse_bad = None
        FA_delta_rmse_bad = None
        FA_delta_0010_rmse_bad = None
        FA_delta_1020_rmse_bad = None   

    # == Step 9 visulize the RMSE/DELTA RMSE of modeled and obs between posterior and prior ==
    # the period is 2000-2010
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= dLdt_delta_0010_rmse_good, delta_rmse_bad= dLdt_delta_0010_rmse_bad, breaks_index = True,
                               bin_width= 0.3,xlim_left =(-50,-42),xlim_right =(-10,1),x_breaks = -10,
                               save_path=postpro_output_fp_region, save_name=None,item_name='dLdt_myr', period='2000-2010')
    # the period is 2010-2020
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= dLdt_delta_1020_rmse_good, delta_rmse_bad= dLdt_delta_1020_rmse_bad, breaks_index = True,
                               bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                               save_path=postpro_output_fp_region, save_name=None,item_name='dLdt_myr', period='2010-2020')
    # the period is 2000-2020
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= dLdt_delta_rmse_good, delta_rmse_bad= dLdt_delta_rmse_bad, breaks_index = True,
                               bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                               save_path=postpro_output_fp_region, save_name=None,item_name='dLdt_myr', period='2000-2020')
    # MB_clim annual weighted prior and posterior and obs,GOOD and BAD
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= MB_clim_delta_0010_rmse_good, delta_rmse_bad= MB_clim_delta_0010_rmse_bad, breaks_index = False,
                               bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                               save_path=postpro_output_fp_region, save_name=None,item_name='MB_clim_mwea', period='2000-2010')
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= MB_clim_delta_1020_rmse_good, delta_rmse_bad= MB_clim_delta_1020_rmse_bad, breaks_index = False,
                                 bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                                 save_path=postpro_output_fp_region, save_name=None,item_name='MB_clim_mwea', period='2010-2020')
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= MB_clim_delta_rmse_good, delta_rmse_bad= MB_clim_delta_rmse_bad, breaks_index = False,
                               bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                               save_path=postpro_output_fp_region, save_name=None,item_name='MB_clim_mwea', period='2000-2020')
    # FA annual weighted prior and posterior and obs, GOOD and BAD
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= FA_delta_0010_rmse_good, delta_rmse_bad= FA_delta_0010_rmse_bad, breaks_index = False,
                               bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                               save_path=postpro_output_fp_region, save_name=None,item_name='FA_gta', period='2000-2010')
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= FA_delta_1020_rmse_good, delta_rmse_bad= FA_delta_1020_rmse_bad, breaks_index = False,
                                 bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                                 save_path=postpro_output_fp_region, save_name=None,item_name='FA_gta', period='2010-2020')
    Vis_ts.plot_delta_rmse_histograms(delta_rmse_good= FA_delta_rmse_good, delta_rmse_bad= FA_delta_rmse_bad, breaks_index = False,
                               bin_width= 0.3,xlim_left =(-60,-51),xlim_right =(-25,20),x_breaks = -25,
                               save_path=postpro_output_fp_region, save_name=None,item_name='FA_gta', period='2000-2020')



    # == Step 10: Visulize the RMSE between modeled and obs for both posterior and prior, and converged and unconverged ==
    # dLdt annual weighted prior and posterior and obs
    # the period is 2000-2010
    Vis_ts.plot_rmse_histograms(rmse_good_prior=dLdt_prior_0010_rmse_good,rmse_good_poster=dLdt_poster_0010_rmse_good,
                                rmse_bad_prior=dLdt_prior_0010_rmse_bad,rmse_bad_poster=dLdt_poster_0010_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='dLdt_myr', period='2000-2010')
    # the period is 2010-2020
    Vis_ts.plot_rmse_histograms(rmse_good_prior=dLdt_prior_1020_rmse_good,rmse_good_poster=dLdt_poster_1020_rmse_good,
                                rmse_bad_prior=dLdt_prior_1020_rmse_bad,rmse_bad_poster=dLdt_poster_1020_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='dLdt_myr', period='2010-2020')
    # the period is 2000-2020
    Vis_ts.plot_rmse_histograms(rmse_good_prior=dLdt_prior_rmse_good,rmse_good_poster=dLdt_poster_rmse_good,
                                rmse_bad_prior=dLdt_prior_rmse_bad,rmse_bad_poster=dLdt_poster_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='dLdt_myr', period='2000-2020')
    # TODO MB_clim annual weighted prior and posterior and obs,GOOD and BAD
    Vis_ts.plot_rmse_histograms(rmse_good_prior=MB_clim_prior_0010_rmse_good,rmse_good_poster=MB_clim_poster_0010_rmse_good,
                                rmse_bad_prior=MB_clim_prior_0010_rmse_bad,rmse_bad_poster=MB_clim_poster_0010_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='MB_clim_mwea', period='2000-2010')
    Vis_ts.plot_rmse_histograms(rmse_good_prior=MB_clim_prior_1020_rmse_good,rmse_good_poster=MB_clim_poster_1020_rmse_good,
                                rmse_bad_prior=MB_clim_prior_1020_rmse_bad,rmse_bad_poster=MB_clim_poster_1020_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='MB_clim_mwea', period='2010-2020')
    Vis_ts.plot_rmse_histograms(rmse_good_prior=MB_clim_prior_rmse_good,rmse_good_poster=MB_clim_poster_rmse_good,
                                rmse_bad_prior=MB_clim_prior_rmse_bad,rmse_bad_poster=MB_clim_poster_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='MB_clim_mwea', period='2000-2020')
    # TODO FA annual weighted prior and posterior and obs, GOOD and BAD
    Vis_ts.plot_rmse_histograms(rmse_good_prior=FA_prior_0010_rmse_good,rmse_good_poster=FA_poster_0010_rmse_good,
                                rmse_bad_prior=FA_prior_0010_rmse_bad,rmse_bad_poster=FA_poster_0010_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='FA_gta', period='2000-2010')
    Vis_ts.plot_rmse_histograms(rmse_good_prior=FA_prior_1020_rmse_good,rmse_good_poster=FA_poster_1020_rmse_good,
                                rmse_bad_prior=FA_prior_1020_rmse_bad,rmse_bad_poster=FA_poster_1020_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='FA_gta', period='2010-2020')
    Vis_ts.plot_rmse_histograms(rmse_good_prior=FA_prior_rmse_good,rmse_good_poster=FA_poster_rmse_good,
                                rmse_bad_prior=FA_prior_rmse_bad,rmse_bad_poster=FA_poster_rmse_bad, bins_width = 0.5,
                         save_path=postpro_output_fp_region, save_name=None, item_name='FA_gta', period='2000-2020')

    


if __name__ == "__main__":
    # Run the main function
    main()
    print("Analysis completed successfully.")



#%% Running example
# python analysis_calibration.py --region_id 7 --data_index 'Annual'