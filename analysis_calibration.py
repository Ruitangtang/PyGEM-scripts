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

    # == Step 1 : Load the model output statistic info data ====
    all_glac_data_stats,mean_data,sum_data = stats_t.read_extract_data_region(region_output_path = model_output_fp_region,
                                                                            region_params_path= model_param_fp_region,reg_id= reg_id,
                                                                            data_index = 'Annual')
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
                                                                       key_value='massbalclim_TMS_model_array_annual_Gta',
                                                                       file_name='mean_MB_clim_1020_raw_gta',period='2010_2020')
    mean_MB_clim_0020_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_Gta',
                                                                       file_name='mean_MB_clim_0020_raw_gta',period='2000_2020')
    mean_MB_clim_0010_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbalclim_TMS_model_array_annual_Gta',
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
                                                                       key_value='massbaltotal_TMS_model_array_annual_Gta',
                                                                       file_name='mean_MB_total_1020_raw_gta',period='2010_2020')
    mean_MB_total_0020_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_Gta',
                                                                       file_name='mean_MB_total_0020_raw_gta',period='2000_2020')
    mean_MB_total_0010_raw_gta = stats_t.extract_and_save_ensemble_data(mean_data,save_path = postpro_output_fp_region,
                                                                       key_value='massbaltotal_TMS_model_array_annual_Gta',
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
        ('mean_stats_20002010_massbalclim_TMS_model_array_annual_Gta', 'mean_0010_mb_clim_Gta'),
        ('mean_stats_20102020_massbalclim_TMS_model_array_annual_Gta', 'mean_1020_mb_clim_Gta'),
        ('mean_stats_20002020_massbalclim_TMS_model_array_annual_Gta', 'mean_0020_mb_clim_Gta'),
        # Sum of mass balance clim GTA statistics
        ('sum_stats_20002010_massbalclim_TMS_model_array_annual_Gta', 'sum_0010_mb_clim_Gt'),
        ('sum_stats_20102020_massbalclim_TMS_model_array_annual_Gta', 'sum_1020_mb_clim_Gt'),
        ('sum_stats_20002020_massbalclim_TMS_model_array_annual_Gta', 'sum_0020_mb_clim_Gt'),

        # Mean mass balance total GTA statistics
        ('mean_stats_20002010_massbaltotal_TMS_model_array_annual_Gta', 'mean_0010_mb_total_Gta'),
        ('mean_stats_20102020_massbaltotal_TMS_model_array_annual_Gta', 'mean_1020_mb_total_Gta'),
        ('mean_stats_20002020_massbaltotal_TMS_model_array_ annual_Gta', 'mean_0020_mb_total_Gta'),
        # Sum of mass balance total GTA statistics
        ('sum_stats_20002010_massbaltotal_TMS_model_array_annual_Gta', 'sum_0010_mb_total_Gt'),
        ('sum_stats_20102020_massbaltotal_TMS_model_array_annual_Gta', 'sum_1020_mb_total_Gt'),
        ('sum_stats_20002020_massbaltotal_TMS_model_array_annual_Gta', 'sum_0020_mb_total_Gt'),

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
    # list all the observation data
    # List all data
    fa_20002010_fp = os.path.join(obs_data_fp,'frontal_ablation_obs_20002010.csv')
    fa_20102020_fp = os.path.join(obs_data_fp,'frontal_ablation_obs_20102020.csv')
    fa_20002020_fp = os.path.join(obs_data_fp,'frontal_ablation_obs_20002020.csv')
    dLdt_20002020_fp = os.path.join(obs_data_fp,'lengthchange_annual_rgi_region01_7_20002020.csv')
    mb_20002010_fp  = os.path.join(obs_data_fp,'mass_balance_obs_20002010.csv')
    mb_20102020_fp  = os.path.join(obs_data_fp,'mass_balance_obs_20102020.csv')
    mb_20002020_fp  = os.path.join(obs_data_fp,'mass_balance_obs_20002020.csv')
    mb_20002020_corr_fp  = os.path.join(obs_data_fp,'mass_balance_obs_20002020_corr.csv')
    fa_obs_20002010 = stats_t.read_data_from_file(fa_20002010_fp)
    fa_obs_20102020 = stats_t.read_data_from_file(fa_20102020_fp)
    fa_obs_20002020 = stats_t.read_data_from_file(fa_20002020_fp)
    mb_obs_20002010 = stats_t.read_data_from_file(mb_20002010_fp)
    mb_obs_20102020 = stats_t.read_data_from_file(mb_20102020_fp)
    mb_obs_20002020 = stats_t.read_data_from_file(mb_20002020_fp)
    mb_obs_20002020_corr = stats_t.read_data_from_file(mb_20002020_corr_fp)
    dLdt_20002020 = stats_t.read_data_from_file(dLdt_20002020_fp)
    # Create the new DataFrame from selected columns
    fa_obs_20002010_gta = pd.DataFrame()
    fa_obs_20002010_gta ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in fa_obs_20002010.RGIId.values]
    fa_obs_20002010_gta ['fa_gta_obs']= fa_obs_20002010['fa_gta_obs']
    fa_obs_20002010_gta ['fa_gta_obs_unc']= fa_obs_20002010['fa_gta_obs_unc']
    fa_obs_20102020_gta = pd.DataFrame()
    fa_obs_20102020_gta ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in fa_obs_20102020.RGIId.values]
    fa_obs_20102020_gta ['fa_gta_obs']= fa_obs_20102020['fa_gta_obs']
    fa_obs_20102020_gta ['fa_gta_obs_unc']= fa_obs_20102020['fa_gta_obs_unc']

    fa_obs_20002020_gta = pd.DataFrame()
    fa_obs_20002020_gta ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in fa_obs_20002020.RGIId.values]
    fa_obs_20002020_gta ['fa_gta_obs']= fa_obs_20002020['fa_gta_obs']
    fa_obs_20002020_gta ['fa_gta_obs_unc']= fa_obs_20002020['fa_gta_obs_unc']

    mb_obs_20002010_mwea = pd.DataFrame()
    mb_obs_20002010_mwea ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in mb_obs_20002010.RGIId.values]
    mb_obs_20002010_mwea ['mb_clim_mwea'] = mb_obs_20002010['mb_clim_mwea']
    mb_obs_20002010_mwea ['mb_clim_mwea_err'] = mb_obs_20002010['mb_clim_mwea_err']

    mb_obs_20102020_mwea = pd.DataFrame()
    mb_obs_20102020_mwea ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in mb_obs_20102020.RGIId.values]
    mb_obs_20102020_mwea ['mb_clim_mwea'] = mb_obs_20102020['mb_clim_mwea']
    mb_obs_20102020_mwea ['mb_clim_mwea_err'] = mb_obs_20102020['mb_clim_mwea_err']

    mb_obs_20002020_mwea = pd.DataFrame()
    mb_obs_20002020_mwea ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in mb_obs_20002020.RGIId.values]
    mb_obs_20002020_mwea ['mb_clim_mwea'] = mb_obs_20002020['mb_clim_mwea']
    mb_obs_20002020_mwea ['mb_clim_mwea_err'] = mb_obs_20002020['mb_clim_mwea_err']

    mb_obs_20002020_mwea_corr = pd.DataFrame()
    mb_obs_20002020_mwea_corr ['rgiid']= [ f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in mb_obs_20002020_corr.RGIId.values]
    mb_obs_20002020_mwea_corr ['mb_clim_mwea'] = mb_obs_20002020_corr['mb_clim_mwea']
    mb_obs_20002020_mwea_corr ['mb_clim_mwea_err'] = mb_obs_20002020_corr['mb_clim_mwea_err']

    # frontal ablation with the unit m w.e. a-1
    fa_obs_20002010_mwea = pd.DataFrame()
    fa_obs_20002010_mwea ['rgiid']= fa_obs_20002010_gta['rgiid']
    fa_obs_20002010_mwea ['fa_mwea_obs']= fa_obs_20002010_gta['fa_gta_obs']*1000./fa_obs_20002010['Area_km2']
    fa_obs_20002010_mwea ['fa_mwea_obs_unc']= fa_obs_20002010_gta['fa_gta_obs_unc']*1000./fa_obs_20002010['Area_km2']
    fa_obs_20102020_mwea = pd.DataFrame()
    fa_obs_20102020_mwea ['rgiid']= fa_obs_20102020_gta['rgiid']
    fa_obs_20102020_mwea ['fa_mwea_obs']= fa_obs_20102020_gta['fa_gta_obs']*1000./fa_obs_20102020['Area_km2']
    fa_obs_20102020_mwea ['fa_mwea_obs_unc']= fa_obs_20102020_gta['fa_gta_obs_unc']*1000./fa_obs_20102020['Area_km2']

    fa_obs_20002020_mwea = pd.DataFrame()
    fa_obs_20002020_mwea ['rgiid']=  fa_obs_20002020_gta['rgiid']
    fa_obs_20002020_mwea ['fa_mwea_obs']= fa_obs_20002020_gta['fa_gta_obs']*1000./fa_obs_20002020['Area_km2']
    fa_obs_20002020_mwea ['fa_mwea_obs_unc']= fa_obs_20002020_gta['fa_gta_obs_unc']*1000./fa_obs_20002020['Area_km2']

    # mass balance with the unit gt a-1
    mb_obs_20002010_gta = pd.DataFrame()
    mb_obs_20002010_gta ['rgiid']= mb_obs_20002010_mwea['rgiid']
    mb_obs_20002010_gta ['mb_clim_gt'] = mb_obs_20002010_mwea['mb_clim_mwea']/1000.*mb_obs_20002010['area']
    mb_obs_20002010_gta ['mb_clim_gt_err'] = mb_obs_20002010_mwea['mb_clim_mwea_err']/1000.*mb_obs_20002010['area']
    mb_obs_20102020_gta = pd.DataFrame()
    mb_obs_20102020_gta ['rgiid']= mb_obs_20102020_mwea['rgiid']
    mb_obs_20102020_gta ['mb_clim_gt'] = mb_obs_20102020_mwea['mb_clim_mwea']/1000.*mb_obs_20102020['area']
    mb_obs_20102020_gta ['mb_clim_gt_err'] = mb_obs_20102020_mwea['mb_clim_mwea_err']/1000.*mb_obs_20102020['area']
    mb_obs_20002020_gta = pd.DataFrame()
    mb_obs_20002020_gta ['rgiid']= mb_obs_20002020_mwea['rgiid']
    mb_obs_20002020_gta ['mb_clim_gt'] = mb_obs_20002020_mwea['mb_clim_mwea']/1000.*mb_obs_20002020['area']
    mb_obs_20002020_gta ['mb_clim_gt_err'] = mb_obs_20002020_mwea['mb_clim_mwea_err']/1000.*mb_obs_20002020['area']

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
    rgiid_good = rgiid_good.sort_values('rgiid')
    rgiid_bad = stats_t.extract_rgi_ids(filepath=AMIS_statis_fp_region,filename='Bad_AMIS.txt')
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
                            obs_name = 'mb_clim_gt' , obs_unc_name = 'mb_clim_gt_err', modeled_key = 'massbalclim_Gta_TMS_model_array_annual', item_name = 'Climatic mass balance (Gt a$^{-1}$)',
                            period = '2010-2020',reg_id = reg_id,Xlabel = 'Climatic mass balance (Gt a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (Gt a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20002010 gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20002010_gta, modeled_df = mean_0010_mb_clim_Gta, modeled_df_raw= mean_MB_clim_0010_raw_gta,
                            obs_name = 'mb_clim_gt' , obs_unc_name = 'mb_clim_gt_err', modeled_key = 'massbalclim_Gta_TMS_model_array_annual', item_name = 'Climatic mass balance (Gt a$^{-1}$)',
                            period = '2000-2010',reg_id = reg_id,Xlabel = 'Climatic mass balance (Gt a$^{-1}$, observed)',Ylabel = 'Climatic mass balance (Gt a$^{-1}$, modeled)', Xlim = (-3.5,4), Ylim = (-3.5,4),
                            subplot_label_L ='c', subplot_label_R = 'd',title= None, save_path = postpro_output_fp_region, save_name=None,Good_AMIS = rgiid_good,Bad_AMIS = rgiid_bad,
                                 legend_index = False,logx=False,logy=False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = (0,0.1), zoom_ylim = (0,0.1),zoom_position = [0.68, 0.68, 0.28, 0.28],zoom_ticklabels = True)
    # MB_clim_20002020 gta
    Vis_ts.plot_cdf_and_one_to_one_Good_Bad_All_Inset(observed_df = mb_obs_20002020_gta, modeled_df = mean_0020_mb_clim_Gta, modeled_df_raw= mean_MB_clim_0020_raw_gta,
                            obs_name = 'mb_clim_gt' , obs_unc_name = 'mb_clim_gt_err', modeled_key = 'massbalclim_Gta_TMS_model_array_annual', item_name = 'Climatic mass balance (Gt a$^{-1}$)',
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


if __name__ == "__main__":
    # Run the main function
    main()
    print("Analysis completed successfully.")



#%% Running example
# python analysis_calibration.py --region_id 7 --data_index 'Annual'