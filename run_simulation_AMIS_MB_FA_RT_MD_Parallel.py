"""Run a model simulation."""
# Default climate data is ERA-Interim; specify CMIP6 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.

# Revised by Ruitang Yang on 14 Oct 2024; it's a copy from run_simulation_FA_Rt.py, but with the following changes:
#  revising it to run the future climate data from CMIP6,and adding the option to visulize the results and save them as csv file
# Revised by Ruitang Yang supported by Matvey Debolskiy on 14 Mar 2025; it's a copy from run_simulation_AMIS_MB_FA_RT_Parallel.py, but with the following changes:
# add the fucntion calc_stats_array_Restruct ( Calculate stats for a given variable, but it will restruct the array based on the info from Unique function
# of Model Posterior Parameters) and the function create_xrdataset_all_statis (compared to create_xrdataset, which will include all statistic infomation), and parallel processing
# in the function simu_MB_FA, which will call the function simu_MB_FA_single_glac and process_for_parallel for each glacier in parallel

# Built-in libraries
import argparse
import collections
import copy
import inspect
import multiprocessing
import os
import sys
import time
import cftime
import traceback
import pdb
import json
import warnings

# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation
from multiprocessing import Pool, cpu_count
import xarray as xr
from functools import partial

try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')

# Local libraries
import pygem
import pygem.gcmbiasadj as gcmbiasadj
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.oggm_compat import single_flowline_glacier_directory_with_calving
from pygem.shop import debris 
from pygem import class_climate
import Visualization_timeseries as Visualization_timeseries
import statistic_tool as stats_tl


import oggm
oggm_version = float(oggm.__version__[0:3])
from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
if oggm_version > 1.301:
    from oggm.core.massbalance import apparent_mb_from_any_mb # Newer Version of OGGM
else:
    from oggm.core.climate import apparent_mb_from_any_mb # Older Version of OGGM
from oggm.core.flowline import FluxBasedModel, SemiImplicitModel
from oggm.core.calving_Jan_Ruitang import CalvingFluxBasedModelJanRt
from oggm.core.inversion_RT_New import find_inversion_calving_from_any_mb
#from oggm.core.inversion import find_inversion_calving_from_any_mb
from Visualization_timeseries import plot_timeseries_stats,plot_timeseries_stats_sub

cfg.PARAMS['hydro_month_nh']=1
cfg.PARAMS['hydro_month_sh']=1
cfg.PARAMS['trapezoid_lambdas'] = 1


# ---- Store_monthly_step ----
store_monthly_step = False
mb_elev_feedback = 'annual'
# TODO : Add the option to store monthly step results for mass balance and glacier dynamics
Dynamic_step_Monthly = False



#%% ----- plot save path -----
output_fp_cali = pygem_prms.main_directory + '/Calibration/'
output_fp = pygem_prms.main_directory + '/Simulation/'
os.makedirs(output_fp, exist_ok=True)
save_path_figure = output_fp + '/figures/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_figure):
    os.makedirs(save_path_figure)
save_path_parameter = output_fp + '/parameter/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_parameter):
    os.makedirs(save_path_parameter)

save_path_modeloutput = output_fp + '/modeloutput/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_modeloutput):
    os.makedirs(save_path_modeloutput)


# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    gcm_list_fn (optional) : str
        text file that contains the climate data to be used in the model simulation
    gcm_name (optional) : str
        gcm name
    scenario (optional) : str
        representative concentration pathway or shared socioeconomic pathway (ex. 'rcp26', 'ssp585')
    realization (optional) : str
        single realization from large ensemble (ex. '1011.001', '1301.020')
        see CESM2 Large Ensemble Community Project by NCAR for more information
    realization_list (optional) : str
        text file that contains the realizations to be used in the model simulation
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_region01 (optional) : int
        Randolph Glacier Inventory region number (01-19)
    rgi_glac_number (optional) : str
        Randolph Glacier Inventory glacier number (ex. '1.00001')
        if None, then all glaciers in the region will be used
    rgi_glac_number_fn (optional) : str
        filename of .pkl/json file containing a list of glacier numbers that used to run batches on the supercomputer
    batch_number (optional): int
        batch number used to differentiate output on supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    debug (optional) : int
        Switch for turning debug printing on or off (default = 0 (off))
    debug_spc (optional) : int
        Switch for turning debug printing of spc on or off (default = 0 (off))
    hugonnet_fn (optional) : str
        filename of .pkl/.csv file containing period averaged mass balance (climatic) data (observations)

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-rgi_region01', type=int, default=None,
                        help='Randoph Glacier Inventory region')
    parser.add_argument('-rgi_glac_number', type=str, default=None,
                        help='Randoph Glacier Inventory region')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-scenario', action='store', type=str, default=None,
                        help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)')
    parser.add_argument('-realization', action='store', type=str, default=None,
                        help='realization from large ensemble used for model run (ex. 1011.001 or 1301.020)')
    parser.add_argument('-realization_list', action='store', type=str, default=None,
                        help='text file full of realizations to run')
    parser.add_argument('-gcm_bc_startyear', action='store', type=int, default=pygem_prms.gcm_bc_startyear,
                        help='start year for bias correction')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms.gcm_startyear,
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms.gcm_endyear,
                        help='start year for the model run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-batch_number', action='store', type=int, default=None,
                        help='Batch number used to differentiate output on supercomputer')
    parser.add_argument('-modelprms_fp', action='store', type=str, default=None,
                    help='model parameters filepath')
    parser.add_argument('-hugonnet_fn', action='store', type=str, default=None,
                    help='Filename containing period averaged mass balance (climatic) data (observations)')
    # flags
    parser.add_argument('-option_ordered', action='store_true',
                        help='Flag to keep glacier lists ordered (default is off)')
    parser.add_argument('-option_parallels', action='store_true',
                        help='Flag to use or not use parallels (default is off)')
    parser.add_argument('-debug', action='store_true',
                        help='Flag for debugging (default is off')
    parser.add_argument('-debug_spc', action='store_true',
                        help='Flag for debugging (default is off')


    return parser


def calc_stats_array(data, stats_cns=pygem_prms.sim_stat_cns):
    """
    Calculate stats for a given variable

    Parameters
    ----------
    vn : str
        variable name
    ds : xarray dataset
        dataset of output with all ensemble simulations

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    stats = None
    if 'mean' in stats_cns:
        if stats is None:
            stats = np.nanmean(data,axis=1)[:,np.newaxis]
    if 'std' in stats_cns:
        stats = np.append(stats, np.nanstd(data,axis=1)[:,np.newaxis], axis=1)
    if '2.5%' in stats_cns:
        stats = np.append(stats, np.nanpercentile(data, 2.5, axis=1)[:,np.newaxis], axis=1)
    if '25%' in stats_cns:
        stats = np.append(stats, np.nanpercentile(data, 25, axis=1)[:,np.newaxis], axis=1)
    if 'median' in stats_cns:
        if stats is None:
            stats = np.nanmedian(data, axis=1)[:,np.newaxis]
        else:
            stats = np.append(stats, np.nanmedian(data, axis=1)[:,np.newaxis], axis=1)
    if '75%' in stats_cns:
        stats = np.append(stats, np.nanpercentile(data, 75, axis=1)[:,np.newaxis], axis=1)
    if '97.5%' in stats_cns:
        stats = np.append(stats, np.nanpercentile(data, 97.5, axis=1)[:,np.newaxis], axis=1)
    if 'mad' in stats_cns:
        stats = np.append(stats, median_abs_deviation(data, axis=1, nan_policy='omit')[:,np.newaxis], axis=1)
    return stats


def calc_stats_array_Restruct(data_uniq, stats_cns=pygem_prms.sim_stat_cns, rgiid_ind =None,reg_id=None,glacier_id=None,
                              object_name=None,traceback_fp=None,warning_fp=None):
    """
    Calculate stats for a given variable, but it will restruct the array based on the info from Unique function
    of Model Posterior Parameters

    Parameters
    ----------
    data_uniq : xarray dataset
        dataset of output with all ensemble simulations / unique ensemble
    stats_cns : list
        list of statistics to calculate, e.g. ['mean', 'std', '2.5%', '25%', 'median', '75%', '97.5%', 'mad']
    rgiid_ind : str
        is the glacier id
    reg_id : str
        is the region id, e.g. '01' for RGI region 01
    glacier_id : str
        is the glacier id, which is used to get the unique info of Model posterior parameters, e.g. '7.00125'
    object_name : str
        is the name of the object being processed, e.g. 'glac_length_monthly'
    traceback_fp : str
        is the file path for the traceback information
    warning_fp : str
        is the file path for the warning information

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    #%% Restruct the array
    # ==== read the Unique info of Model posterior parameters
    output_folder_post_params_unique = os.path.join(output_fp_cali,'parameter',reg_id,glacier_id,'Poster','Unique')
    output_filename_params_unique_Info = f'calibration_poster_Params_unique_{rgiid_ind}_Info.json'
    output_fp_params_unique_Info = os.path.join(output_folder_post_params_unique, output_filename_params_unique_Info) 
    
    with open(output_fp_params_unique_Info, 'r') as f:
        parms_UniqInfo_dict = json.load(f)
    unique_counts= parms_UniqInfo_dict['unique_counts']
    
    data = np.repeat(data_uniq,unique_counts,axis = 1)
    #%% do the stats
    stats = None

    # Set up warning capture to use the custom handler
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: stats_tl.custom_warning_handler(
        message, category, filename, lineno, object_name=object_name, file_fp=warning_fp
    )

    warnings.simplefilter("always", RuntimeWarning)  # Capture all RuntimeWarnings
    # Now calculate the mean safely
    try:
        if 'mean' in stats_cns:
            try:
                if stats is None:
                    stats = np.nanmean(data, axis=1)[:, np.newaxis]
                else:
                    stats = np.append(stats, np.nanmean(data, axis=1)[:, np.newaxis], axis=1)
            except Exception as e:
                # log the traceback information to the traceback file
                stats_tl.log_traceback(object_name=object_name, traceback_fp=traceback_fp)
        if 'mad' in stats_cns:
            stats = np.append(stats, median_abs_deviation(data, axis=1, nan_policy='omit')[:,np.newaxis], axis=1)
        if '2.5%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 2.5, axis=1)[:,np.newaxis], axis=1)
        if '25%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 25, axis=1)[:,np.newaxis], axis=1)
        if 'median' in stats_cns:
            if stats is None:
                stats = np.nanmedian(data, axis=1)[:,np.newaxis]
            else:
                stats = np.append(stats, np.nanmedian(data, axis=1)[:,np.newaxis], axis=1)
        if '75%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 75, axis=1)[:,np.newaxis], axis=1)
        if '97.5%' in stats_cns:
            stats = np.append(stats, np.nanpercentile(data, 97.5, axis=1)[:,np.newaxis], axis=1)
        if 'std' in stats_cns:
            stats = np.append(stats, np.nanstd(data,axis=1)[:,np.newaxis], axis=1)
    except Exception as e:
        # log the traceback information to the traceback file
        stats_tl.log_traceback(object_name=object_name, traceback_fp=traceback_fp)

    return stats


def create_xrdataset(glacier_rgi_table, dates_table, option_wateryear=pygem_prms.gcm_wateryear, 
                     export_extra_vars=pygem_prms.export_extra_vars):
    """
    Create empty xarray dataset that will be used to record simulation runs.

    Parameters
    ----------
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    dates_table : pandas dataframe
        table of the dates, months, days in month, etc.

    Returns
    -------
    output_ds_all : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # Create empty datasets for each variable and merge them
    # Coordinate values
    glac_values = np.array([glacier_rgi_table.name])

    # Time attributes and values
    if option_wateryear == 'hydro':
        year_type = 'water year'
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'calendar':
        year_type = 'calendar year'
        annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'custom':
        year_type = 'custom year'
       
    time_values = dates_table.loc[pygem_prms.gcm_spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
    time_values = [cftime.DatetimeNoLeap(x.year, x.month, x.day) for x in time_values]

    # append additional year to year_values to account for mass and area at end of period
    year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
    year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))

    # Variable coordinates dictionary
    output_coords_dict = collections.OrderedDict()
    output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['CenLon'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['CenLat'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['O1Region'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['O2Region'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['Area'] = collections.OrderedDict([('glac', glac_values)])
    
    output_coords_dict['glac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                         ('time', time_values)]) 
    output_coords_dict['glac_area_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                      ('year', year_values)])
    output_coords_dict['glac_length_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                        ('year', year_values)])
    output_coords_dict['glac_length_change_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                              ('year', year_values)])
    output_coords_dict['glac_frontalablation_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                                 ('year', year_values)])
    output_coords_dict['glac_massbalclim_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                              ('year', year_values)])
    output_coords_dict['glac_massbaltotal_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                              ('year', year_values)])
    output_coords_dict['glac_mass_annual'] = collections.OrderedDict([('glac', glac_values), 
                                                                        ('year', year_values)])
    output_coords_dict['glac_mass_bsl_annual'] = collections.OrderedDict([('glac', glac_values), 
                                                                            ('year', year_values)])
    output_coords_dict['glac_ELA_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                     ('year', year_values)])
    output_coords_dict['offglac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                            ('time', time_values)])
    if pygem_prms.sim_iters > 1:
        output_coords_dict['glac_runoff_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                 ('time', time_values)])
        output_coords_dict['glac_area_annual_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('year', year_values)])
        output_coords_dict['glac_length_annual_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                 ('year', year_values)])
        output_coords_dict['glac_length_change_annual_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                       ('year', year_values)])
        output_coords_dict['glac_frontalablation_annual_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                          ('year', year_values)])
        output_coords_dict['glac_massbalclim_annual_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                     ('year', year_values)])
        output_coords_dict['glac_massbaltotal_annual_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                     ('year', year_values)])
        output_coords_dict['glac_mass_annual_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                ('year', year_values)])
        output_coords_dict['glac_mass_bsl_annual_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                    ('year', year_values)])
        output_coords_dict['glac_ELA_annual_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                             ('year', year_values)])
        output_coords_dict['offglac_runoff_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                    ('time', time_values)])
        
    if export_extra_vars:
        output_coords_dict['glac_prec_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                           ('time', time_values)])
        output_coords_dict['glac_temp_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                           ('time', time_values)])
        output_coords_dict['glac_acc_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                          ('time', time_values)])
        output_coords_dict['glac_refreeze_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_melt_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                           ('time', time_values)])
        output_coords_dict['glac_frontalablation_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                      ('time', time_values)])
        output_coords_dict['glac_length_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                       ('time', time_values)])
        output_coords_dict['glac_length_change_monthly'] = collections.OrderedDict([('glac', glac_values),
                                                                                       ('time', time_values)])
        output_coords_dict['glac_massbalclim_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
        output_coords_dict['glac_massbaltotal_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
        output_coords_dict['glac_snowline_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_mass_change_ignored_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                                       ('year', year_values)])
        output_coords_dict['offglac_prec_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('time', time_values)])
        output_coords_dict['offglac_refreeze_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
        output_coords_dict['offglac_melt_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('time', time_values)])
        output_coords_dict['offglac_snowpack_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
        if pygem_prms.sim_iters > 1:
            output_coords_dict['glac_prec_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
            output_coords_dict['glac_temp_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
            output_coords_dict['glac_acc_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
            output_coords_dict['glac_refreeze_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                       ('time', time_values)])
            output_coords_dict['glac_melt_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
            output_coords_dict['glac_frontalablation_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                              ('time', time_values)])
            output_coords_dict['glac_length_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                              ('time', time_values)])
            output_coords_dict['glac_length_change_monthly_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                             ('time', time_values)])
            output_coords_dict['glac_massbaltotal_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                           ('time', time_values)])
            output_coords_dict['glac_massbalclim_monthly_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                            ('time', time_values)])
            output_coords_dict['glac_snowline_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                       ('time', time_values)])
            output_coords_dict['glac_mass_change_ignored_annual_mad'] = collections.OrderedDict([('glac', glac_values),
                                                                                                   ('year', year_values)])
            output_coords_dict['offglac_prec_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                      ('time', time_values)])
            output_coords_dict['offglac_refreeze_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                          ('time', time_values)])
            output_coords_dict['offglac_melt_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                      ('time', time_values)])
            output_coords_dict['offglac_snowpack_monthly_mad'] = collections.OrderedDict([('glac', glac_values), 
                                                                                          ('time', time_values)])
    
    # Attributes dictionary
    output_attrs_dict = {
        'time': {
                'long_name': 'time',
                 'year_type':year_type,
                 'comment':'start of the month'},
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'year': {
                'long_name': 'years',
                 'year_type': year_type,
                 'comment': 'years referring to the start of each year'},
        'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.0'},
        'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.0'},
        'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.0'},
        'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.0'},
        'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.0'},
        'Area': {
                'long_name': 'glacier area',
                'units': 'm2',
                'comment': 'value from RGIv6.0'},
        'glac_runoff_monthly': {
                'long_name': 'glacier-wide runoff',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'runoff from the glacier terminus, which moves over time'},
        'glac_area_annual': {
                'long_name': 'glacier area',
                'units': 'm2',
                'temporal_resolution': 'annual',
                'comment': 'area at start of the year'},
        'glac_length_annual': {
                'long_name': 'glacier length',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'length at start of the year'},
        'glac_length_change_annual': {
                'long_name': 'glacier length change',
                'units':'m /a',
                'temporal_resolution': 'annual',
                'comment': 'change in length over the year'},
        'glac_frontalablation_annual': {
                'long_name': 'glacier-wide frontal ablation, in water equivalent',
                'units': 'm3',
                'temporal_resolution': 'annual',
                'comment': 'mass losses from calving, subaerial frontal melting, sublimation above the '
                           'waterline and subaqueous frontal melting below the waterline'},
        'glac_massbalclim_annual': {
                'long_name': 'glacier-wide climatic mass balance, in water equivalent',
                'units': 'm3',
                'temporal_resolution': 'annual',
                'comment': 'climatic mass balance is the sum of solid precipitation, refreeze, and melt'},
        'glac_massbaltotal_annual': {
                'long_name': 'glacier-wide total mass balance, in water equivalent',
                'units': 'm3',
                'temporal_resolution': 'annual',
                'comment': 'total mass balance is the sum of climatic mass balance and frontal ablation'},
        'glac_mass_annual': {
                'long_name': 'glacier mass',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'mass of ice based on area and ice thickness at start of the year'},
        'glac_mass_bsl_annual': {
                'long_name': 'glacier mass below sea level',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'mass of ice below sea level based on area and ice thickness at start of the year'},
        'glac_ELA_annual': {
                'long_name': 'annual equilibrium line altitude above mean sea level',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
        'offglac_runoff_monthly': {
                'long_name': 'off-glacier-wide runoff',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'off-glacier runoff from area where glacier no longer exists'},
        }
    
    if pygem_prms.sim_iters > 1:
        output_attrs_dict_mad = {
            'glac_runoff_monthly_mad': {
                    'long_name': 'glacier-wide runoff median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'runoff from the glacier terminus, which moves over time'},
            'glac_area_annual_mad': {
                    'long_name': 'glacier area median absolute deviation',
                    'units': 'm2',
                    'temporal_resolution': 'annual',
                    'comment': 'area at start of the year'},
            'glac_length_annual_mad': {
                    'long_name': 'glacier length median absolute deviation',
                    'units': 'm',
                    'temporal_resolution': 'annual',
                    'comment': 'length at start of the year'},
            'glac_length_change_annual_mad': {
                    'long_name': 'glacier length change rate median absolute deviation',
                    'units': 'm/a',
                    'temporal_resolution': 'annual',
                    'comment': 'length change rate'},
            'glac_frontalablation_annual_mad': {
                    'long_name': 'glacier-wide frontal ablation median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'mass losses from calving, subaerial frontal melting, sublimation above the '
                               'waterline and subaqueous frontal melting below the waterline'},
            'glac_massbalclim_annual_mad': {
                    'long_name': 'glacier-wide climatic mass balance median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'climatic mass balance is the sum of solid precipitation, refreeze, and melt'},
            'glac_massbaltotal_annual_mad': {
                    'long_name': 'glacier-wide total mass balance median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'total mass balance is the sum of climatic mass balance and frontal ablation'},
            'glac_mass_annual_mad': {
                    'long_name': 'glacier mass median absolute deviation',
                    'units': 'kg',
                    'temporal_resolution': 'annual',
                    'comment': 'mass of ice based on area and ice thickness at start of the year'},
            'glac_mass_bsl_annual_mad': {
                    'long_name': 'glacier mass below sea level median absolute deviation',
                    'units': 'kg',
                    'temporal_resolution': 'annual',
                    'comment': 'mass of ice below sea level based on area and ice thickness at start of the year'},
            'glac_ELA_annual_mad': {
                    'long_name': 'annual equilibrium line altitude above mean sea level median absolute deviation',
                    'units': 'm',
                    'temporal_resolution': 'annual',
                    'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
            'offglac_runoff_monthly_mad': {
                    'long_name': 'off-glacier-wide runoff median absolute deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'off-glacier runoff from area where glacier no longer exists'},
            }
        output_attrs_dict.update(output_attrs_dict_mad)
        
    if export_extra_vars:
        output_attrs_dict_extras = {
            'glac_temp_monthly': {
                    'standard_name': 'air_temperature',
                    'long_name': 'glacier-wide mean air temperature',
                    'units': 'K',
                    'temporal_resolution': 'monthly',
                    'comment': ('each elevation bin is weighted equally to compute the mean temperature, and '
                                'bins where the glacier no longer exists due to retreat have been removed')},
            'glac_prec_monthly': {
                    'long_name': 'glacier-wide precipitation (liquid)',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only the liquid precipitation, solid precipitation excluded'},
            'glac_acc_monthly': {
                    'long_name': 'glacier-wide accumulation, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only the solid precipitation'},
            'glac_refreeze_monthly': {
                    'long_name': 'glacier-wide refreeze, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly'},
            'glac_melt_monthly': {
                    'long_name': 'glacier-wide melt, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly'},
            'glac_frontalablation_monthly': {
                    'long_name': 'glacier-wide frontal ablation, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': (
                            'mass losses from calving, subaerial frontal melting, sublimation above the '
                            'waterline and subaqueous frontal melting below the waterline; positive values indicate mass lost like melt')},
            'glac_length_monthly': {
                    'long_name': 'glacier length',
                    'units': 'm ',
                    'temporal_resolution': 'monthly',
                    'comment': (
                            'length')},
            'glac_length_change_monthly': {
                    'long_name': 'glacier length change rate',
                    'units': 'm/a',
                    'temporal_resolution': 'monthly',
                    'comment': (
                            'length change')},
            'glac_massbaltotal_monthly': {
                    'long_name': 'glacier-wide total mass balance, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
            'glac_massbalclim_monthly': {
                'long_name': 'glacier-wide climatic mass balance, in water equivalent',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'climatic mass balance is the sum of solid precipitation, refreeze, and melt'},
            'glac_snowline_monthly': {
                'long_name': 'transient snowline altitude above mean sea level',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'transient snowline is altitude separating snow from ice/firn'},
            'glac_mass_change_ignored_annual': { 
                'long_name': 'glacier mass change ignored',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'glacier mass change ignored due to flux divergence'},
            'offglac_prec_monthly': {
                'long_name': 'off-glacier-wide precipitation (liquid)',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'only the liquid precipitation, solid precipitation excluded'},
            'offglac_refreeze_monthly': {
                    'long_name': 'off-glacier-wide refreeze, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly'},
            'offglac_melt_monthly': {
                    'long_name': 'off-glacier-wide melt, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only melt of snow and refreeze since off-glacier'},
            'offglac_snowpack_monthly': {
                'long_name': 'off-glacier-wide snowpack, in water equivalent',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'}
            }
        output_attrs_dict.update(output_attrs_dict_extras)
        
        if pygem_prms.sim_iters > 1:
            output_attrs_dict_extras_mad = {
                'glac_temp_monthly_mad': {
                        'standard_name': 'air_temperature',
                        'long_name': 'glacier-wide mean air temperature median absolute deviation',
                        'units': 'K',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'each elevation bin is weighted equally to compute the mean temperature, and '
                                'bins where the glacier no longer exists due to retreat have been removed')},
                'glac_prec_monthly_mad': {
                        'long_name': 'glacier-wide precipitation (liquid) median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the liquid precipitation, solid precipitation excluded'},
                'glac_acc_monthly_mad': {
                        'long_name': 'glacier-wide accumulation, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the solid precipitation'},
                'glac_refreeze_monthly_mad': {
                        'long_name': 'glacier-wide refreeze, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly'},
                'glac_melt_monthly_mad': {
                        'long_name': 'glacier-wide melt, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly'},
                'glac_frontalablation_monthly_mad': {
                        'long_name': 'glacier-wide frontal ablation, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'mass losses from calving, subaerial frontal melting, sublimation above the '
                                'waterline and subaqueous frontal melting below the waterline')},
                'glac_length_monthly_mad': {
                        'long_name': 'glacier length ,  median absolute deviation',
                        'units': 'm',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'glacier length ')},
                'glac_length_change_monthly_mad': {
                        'long_name': 'glacier length change rate,median absolute deviation',
                        'units': 'm/a',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'length change')},
                'glac_massbaltotal_monthly_mad': {
                        'long_name': 'glacier-wide total mass balance, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
                'glac_massbalclim_monthly_mad': {
                        'long_name': 'glacier-wide climatic mass balance, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'climatic mass balance is the sum of solid precipitation, refreeze, and melt'},
                'glac_snowline_monthly_mad': {
                        'long_name': 'transient snowline above mean sea level median absolute deviation',
                        'units': 'm',
                        'temporal_resolution': 'monthly',
                        'comment': 'transient snowline is altitude separating snow from ice/firn'},
                'glac_mass_change_ignored_annual_mad': { 
                        'long_name': 'glacier mass change ignored median absolute deviation',
                        'units': 'kg',
                        'temporal_resolution': 'annual',
                        'comment': 'glacier mass change ignored due to flux divergence'},
                'offglac_prec_monthly_mad': {
                        'long_name': 'off-glacier-wide precipitation (liquid) median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the liquid precipitation, solid precipitation excluded'},
                'offglac_refreeze_monthly_mad': {
                        'long_name': 'off-glacier-wide refreeze, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly'},
                'offglac_melt_monthly_mad': {
                        'long_name': 'off-glacier-wide melt, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only melt of snow and refreeze since off-glacier'},
                
                'offglac_snowpack_monthly_mad': {
                        'long_name': 'off-glacier-wide snowpack, in water equivalent, median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'},
                }
            output_attrs_dict.update(output_attrs_dict_extras_mad)
       
    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        count_vn += 1
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    noencoding_vn = ['RGIId']
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
       
        if vn not in noencoding_vn:
            encoding[vn] = {'_FillValue': None,
                            'zlib':True,
                            'complevel':9
                            }
    output_ds_all['RGIId'].values = np.array([glacier_rgi_table.loc['RGIId']])
    output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
    output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
    output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
    output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
    output_ds_all['Area'].values = np.array([glacier_rgi_table.Area * 1e6])
   
    output_ds_all.attrs = {'source': f'PyGEMv{pygem.__version__}',
                       'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                       'history': 'Created by David Rounce (drounce@alaska.edu) on ' + pygem_prms.model_run_date,
                       'update': 'the Frontal ablation and length change added by Ruitang Yang (tangruiyang123@gmail.com) on 2021-06-01 and run on '+pygem_prms.model_run_date,
                       'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91;' + ' doi.org/10.1029/2020GL090213'}
       
    return output_ds_all, encoding


def create_xrdataset_all_statis(glacier_rgi_table, dates_table, option_wateryear=pygem_prms.gcm_wateryear, 
                     export_extra_vars=pygem_prms.export_extra_vars):
    """
    Create empty xarray dataset that will be used to record simulation runs with all statistics.
    Includes mean, std, percentiles (2.5%, 25%, 50%, 75%, 97.5%), and median absolute deviation (mad).

    Parameters
    ----------
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    dates_table : pandas dataframe
        table of the dates, months, days in month, etc.

    Returns
    -------
    output_ds_all : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # Create empty datasets for each variable and merge them
    # Coordinate values
    glac_values = np.array([glacier_rgi_table.name])

    # Time attributes and values
    if option_wateryear == 'hydro':
        year_type = 'water year'
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'calendar':
        year_type = 'calendar year'
        annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'custom':
        year_type = 'custom year'
       
    time_values = dates_table.loc[pygem_prms.gcm_spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
    time_values = [cftime.DatetimeNoLeap(x.year, x.month, x.day) for x in time_values]

    # append additional year to year_values to account for mass and area at end of period
    year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
    year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))

    # Variable coordinates dictionary
    output_coords_dict = collections.OrderedDict()
    output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['CenLon'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['CenLat'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['O1Region'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['O2Region'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['Area'] = collections.OrderedDict([('glac', glac_values)])
    
    # List of all variables that will have statistics
    stat_variables = [
        'glac_runoff_monthly',
        'glac_area_annual',
        'glac_length_annual',
        'glac_length_change_annual',
        'glac_frontalablation_annual',
        'glac_massbalclim_annual',
        'glac_massbaltotal_annual',
        'glac_mass_annual',
        'glac_mass_bsl_annual',
        'glac_ELA_annual',
        'offglac_runoff_monthly'
    ]
    
    if export_extra_vars:
        extra_stat_variables = [
            'glac_prec_monthly',
            'glac_temp_monthly',
            'glac_acc_monthly',
            'glac_refreeze_monthly',
            'glac_melt_monthly',
            'glac_frontalablation_monthly',
            'glac_length_monthly',
            'glac_length_change_monthly',
            'glac_massbaltotal_monthly',
            'glac_massbalclim_monthly',
            'glac_snowline_monthly',
            'glac_mass_change_ignored_annual',
            'offglac_prec_monthly',
            'offglac_refreeze_monthly',
            'offglac_melt_monthly',
            'offglac_snowpack_monthly'
        ]
        stat_variables.extend(extra_stat_variables)
    
    # Add all statistics for each variable
    for vn in stat_variables:
        # Determine if it's monthly or annual
        if 'monthly' in vn:
            time_dim = 'time'
            time_vals = time_values
        else:
            time_dim = 'year'
            time_vals = year_values
            
        # Add coordinates for all statistics
        output_coords_dict[vn + '_mean'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_std'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_p2_5'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_p25'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_median'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_p75'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_p97_5'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
        output_coords_dict[vn + '_mad'] = collections.OrderedDict([('glac', glac_values), (time_dim, time_vals)])
    
    # Attributes dictionary
    output_attrs_dict = {
        'time': {
                'long_name': 'time',
                 'year_type':year_type,
                 'comment':'start of the month'},
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'year': {
                'long_name': 'years',
                 'year_type': year_type,
                 'comment': 'years referring to the start of each year'},
        'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.0'},
        'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.0'},
        'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.0'},
        'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.0'},
        'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.0'},
        'Area': {
                'long_name': 'glacier area',
                'units': 'm2',
                'comment': 'value from RGIv6.0'},
    }
    
    # Base attributes for each variable type
    base_attrs = {
        'glac_runoff': {
            'long_name': 'glacier-wide runoff',
            'units': 'm3',
            'comment': 'runoff from the glacier terminus, which moves over time'},
        'glac_area': {
            'long_name': 'glacier area',
            'units': 'm2',
            'comment': 'area at start of the year'},
        'glac_length': {
            'long_name': 'glacier length',
            'units': 'm',
            'comment': 'length at start of the year'},
        'glac_length_change': {
            'long_name': 'glacier length change',
            'units':'m /a',
            'comment': 'change in length over the year'},
        'glac_frontalablation': {
            'long_name': 'glacier-wide frontal ablation',
            'units': 'm3',
            'comment': 'mass losses from calving, subaerial frontal melting, sublimation above the '
                       'waterline and subaqueous frontal melting below the waterline'},
        'glac_mass': {
            'long_name': 'glacier mass',
            'units': 'kg',
            'comment': 'mass of ice based on area and ice thickness at start of the year'},
        'glac_mass_bsl': {
            'long_name': 'glacier mass below sea level',
            'units': 'kg',
            'comment': 'mass of ice below sea level based on area and ice thickness at start of the year'},
        'glac_ELA': {
            'long_name': 'annual equilibrium line altitude above mean sea level',
            'units': 'm',
            'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
        'offglac_runoff': {
            'long_name': 'off-glacier-wide runoff',
            'units': 'm3',
            'comment': 'off-glacier runoff from area where glacier no longer exists'},
        'glac_temp': {
            'standard_name': 'air_temperature',
            'long_name': 'glacier-wide mean air temperature',
            'units': 'K',
            'comment': ('each elevation bin is weighted equally to compute the mean temperature, and '
                        'bins where the glacier no longer exists due to retreat have been removed')},
        'glac_prec': {
            'long_name': 'glacier-wide precipitation (liquid)',
            'units': 'm3',
            'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'glac_acc': {
            'long_name': 'glacier-wide accumulation, in water equivalent',
            'units': 'm3',
            'comment': 'only the solid precipitation'},
        'glac_refreeze': {
            'long_name': 'glacier-wide refreeze, in water equivalent',
            'units': 'm3'},
        'glac_melt': {
            'long_name': 'glacier-wide melt, in water equivalent',
            'units': 'm3'},
        'glac_frontalablation_monthly': {
            'long_name': 'glacier-wide frontal ablation, in water equivalent',
            'units': 'm3',
            'comment': ('mass losses from calving, subaerial frontal melting, sublimation above the '
                       'waterline and subaqueous frontal melting below the waterline; positive values indicate mass lost like melt')},
        'glac_length_monthly': {
            'long_name': 'glacier length',
            'units': 'm',
            'comment': 'length'},
        'glac_length_change_monthly': {
            'long_name': 'glacier length change rate',
            'units': 'm/a',
            'comment': 'length change'},
        'glac_massbaltotal': {
            'long_name': 'glacier-wide total mass balance, in water equivalent',
            'units': 'm3',
            'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
        'glac_snowline': {
            'long_name': 'transient snowline altitude above mean sea level',
            'units': 'm',
            'comment': 'transient snowline is altitude separating snow from ice/firn'},
        'glac_mass_change_ignored': { 
            'long_name': 'glacier mass change ignored',
            'units': 'kg',
            'comment': 'glacier mass change ignored due to flux divergence'},
        'offglac_prec': {
            'long_name': 'off-glacier-wide precipitation (liquid)',
            'units': 'm3',
            'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'offglac_refreeze': {
            'long_name': 'off-glacier-wide refreeze, in water equivalent',
            'units': 'm3'},
        'offglac_melt': {
            'long_name': 'off-glacier-wide melt, in water equivalent',
            'units': 'm3',
            'comment': 'only melt of snow and refreeze since off-glacier'},
        'offglac_snowpack': {
            'long_name': 'off-glacier-wide snowpack, in water equivalent',
            'units': 'm3',
            'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'}
    }
    
    # Add temporal resolution to attributes
    for vn in stat_variables:
        base_name = vn.replace('_monthly', '').replace('_annual', '')
        if base_name in base_attrs:
            attrs = base_attrs[base_name].copy()
            if 'monthly' in vn:
                attrs['temporal_resolution'] = 'monthly'
            else:
                attrs['temporal_resolution'] = 'annual'
            
            # Add attributes for each statistic
            output_attrs_dict[vn + '_mean'] = {
                'long_name': attrs['long_name'] + ' (mean)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_std'] = {
                'long_name': attrs['long_name'] + ' (standard deviation)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_p2_5'] = {
                'long_name': attrs['long_name'] + ' (2.5th percentile)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_p25'] = {
                'long_name': attrs['long_name'] + ' (25th percentile)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_median'] = {
                'long_name': attrs['long_name'] + ' (median)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_p75'] = {
                'long_name': attrs['long_name'] + ' (75th percentile)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_p97_5'] = {
                'long_name': attrs['long_name'] + ' (97.5th percentile)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
                
            output_attrs_dict[vn + '_mad'] = {
                'long_name': attrs['long_name'] + ' (median absolute deviation)',
                'units': attrs['units'],
                'temporal_resolution': attrs['temporal_resolution'],
                'comment': attrs.get('comment', '')}
    
    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        count_vn += 1
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    
    noencoding_vn = ['RGIId']
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        if vn not in noencoding_vn:
            encoding[vn] = {'_FillValue': None,
                            'zlib':True,
                            'complevel':9
                            }
    
    # Set static glacier properties
    output_ds_all['RGIId'].values = np.array([glacier_rgi_table.loc['RGIId']])
    output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
    output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
    output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
    output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
    output_ds_all['Area'].values = np.array([glacier_rgi_table.Area * 1e6])
   
    output_ds_all.attrs = {'source': f'PyGEMv{pygem.__version__}',
                       'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                       'history': 'Created by David Rounce (drounce@alaska.edu) on ' + pygem_prms.model_run_date,
                       'update': 'the Frontal ablation and length change added by Ruitang Yang (tangruiyang123@gmail.com) on 2021-06-01 and run on '+pygem_prms.model_run_date,
                       'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91;' + ' doi.org/10.1029/2020GL090213'}
    return output_ds_all, encoding


def save_all_statistics(output_ds_all_stats, stats_dict, pygem_prms, Dynamic_step_Monthly):
    """
    Save all statistical information to the dataset.

    Parameters:
    ----------
    output_ds_all_stats : xarray.Dataset
        Dataset to store statistical values.
    stats_dict : dict
        Dictionary mapping variable names to their computed statistical arrays.
    pygem_prms : object
        Contains simulation parameters and settings.
    Dynamic_step_Monthly : bool
        Whether to include dynamic monthly step variables.
    """

    # Define the corresponding statistical suffixes
    stats_labels = ['mean', 'std', '2.5%', '25%', 'median', '75%', '97.5%', 'mad']
    # Define the mapping of statistic labels to consistent suffixes
    stat_label_map = {
        'mean': 'mean',
        'std': 'std',
        '2.5%': 'p2_5',
        '25%': 'p25',
        'median': 'median',
        '75%': 'p75',
        '97.5%': 'p97_5',
        'mad': 'mad'
    }
    
    # Assign computed statistics to the dataset
    for var_name, stats_array in stats_dict.items():
        for i, stat_label in enumerate(stats_labels):
            if stat_label == 'mad' and pygem_prms.sim_iters <= 1:
                continue  # Skip MAD if there’s only one iteration
            
            # Create variable names dynamically using the mapping, e.g., 'glac_runoff_monthly_median'
            stat_var_name =  f"{var_name}_{stat_label_map[stat_label]}"
            
            # Assign the values to the dataset
            output_ds_all_stats[stat_var_name].values[0, :] = stats_array[:, i]

    print("All statistical values assigned to dataset.")
    return output_ds_all_stats



# def create_xrdataset_essential_sims(glacier_rgi_table, dates_table, option_wateryear=pygem_prms.gcm_wateryear,
#                                     sim_iters=pygem_prms.sim_iters):
#     """
#     Create empty xarray dataset that will be used to record simulation runs.

#     Parameters
#     ----------
#     main_glac_rgi : pandas dataframe
#         dataframe containing relevant rgi glacier information
#     dates_table : pandas dataframe
#         table of the dates, months, days in month, etc.

#     Returns
#     -------
#     output_ds_all : xarray Dataset
#         empty xarray dataset that contains variables and attributes to be filled in by simulation runs
#     encoding : dictionary
#         encoding used with exporting xarray dataset to netcdf
#     """
#     # Create empty datasets for each variable and merge them
#     # Coordinate values
#     glac_values = np.array([glacier_rgi_table.name])

#     # Time attributes and values
#     if option_wateryear == 'hydro':
#         year_type = 'water year'
#         annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
#     elif option_wateryear == 'calendar':
#         year_type = 'calendar year'
#         annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
#     elif option_wateryear == 'custom':
#         year_type = 'custom year'

#     time_values = dates_table.loc[pygem_prms.gcm_spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
#     time_values = [cftime.DatetimeNoLeap(x.year, x.month, x.day) for x in time_values]

#     # append additional year to year_values to account for mass and area at end of period
#     year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
#     year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
    
#     sims = np.arange(sim_iters)

#     # Variable coordinates dictionary
#     output_coords_dict = collections.OrderedDict()
#     output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', glac_values)])
#     output_coords_dict['CenLon'] = collections.OrderedDict([('glac', glac_values)])
#     output_coords_dict['CenLat'] = collections.OrderedDict([('glac', glac_values)])
#     output_coords_dict['O1Region'] = collections.OrderedDict([('glac', glac_values)])
#     output_coords_dict['O2Region'] = collections.OrderedDict([('glac', glac_values)])
#     output_coords_dict['Area'] = collections.OrderedDict([('glac', glac_values)])
#     # annual datasets
#     output_coords_dict['glac_area_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     output_coords_dict['glac_length_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     output_coords_dict['glac_length_change_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     output_coords_dict['glac_frontalablation_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     output_coords_dict['glac_massbalclim_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     output_coords_dict['glac_massbaltotal_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     output_coords_dict['glac_mass_annual'] = (
#             collections.OrderedDict([('glac', glac_values), ('year', year_values), ('sim', sims)]))
#     # monthly datasets
#     output_coords_dict['fixed_runoff_monthly'] = (
#             collections.OrderedDict([('glac', glac_values), ('time', time_values), ('sim', sims)]))
    
#     # Attributes dictionary
#     output_attrs_dict = {
#         'time': {
#                 'long_name': 'time',
#                  'year_type':year_type,
#                  'comment':'start of the month'},
#         'glac': {
#                 'long_name': 'glacier index',
#                  'comment': 'glacier index referring to glaciers properties and model results'},
#         'year': {
#                 'long_name': 'years',
#                  'year_type': year_type,
#                  'comment': 'years referring to the start of each year'},
#         'sim': {
#                 'long_name': 'simulation number',
#                 'comment': 'simulation number referring to the MCMC simulation; otherwise, only 1'},
#         'RGIId': {
#                 'long_name': 'Randolph Glacier Inventory ID',
#                 'comment': 'RGIv6.0'},
#         'CenLon': {
#                 'long_name': 'center longitude',
#                 'units': 'degrees E',
#                 'comment': 'value from RGIv6.0'},
#         'CenLat': {
#                 'long_name': 'center latitude',
#                 'units': 'degrees N',
#                 'comment': 'value from RGIv6.0'},
#         'O1Region': {
#                 'long_name': 'RGI order 1 region',
#                 'comment': 'value from RGIv6.0'},
#         'O2Region': {
#                 'long_name': 'RGI order 2 region',
#                 'comment': 'value from RGIv6.0'},
#         'Area': {
#                 'long_name': 'glacier area',
#                 'units': 'm2',
#                 'comment': 'value from RGIv6.0'},
#         'fixed_runoff_monthly': {
#                 'long_name': 'fixed-gauge glacier runoff',
#                 'units': 'm3',
#                 'temporal_resolution': 'monthly',
#                 'comment': 'runoff assuming a fixed gauge station based on initial glacier area'},
#         'glac_area_annual': {
#                 'long_name': 'glacier area',
#                 'units': 'm2',
#                 'temporal_resolution': 'annual',
#                 'comment': 'area at start of the year'},
#         'glac_length_annual': {
#                 'long_name': 'glacier length',
#                 'units': 'm',
#                 'temporal_resolution': 'annual',
#                 'comment': 'length at start of the year'},
#         'glac_length_change_annual': {
#                 'long_name': 'glacier length change',
#                 'units': 'm',
#                 'temporal_resolution': 'annual',
#                 'comment': 'length change from start to end of the year'},
#         'glac_frontalablation_annual': {
#                 'long_name': 'glacier frontal ablation',
#                 'units': 'm3',
#                 'temporal_resolution': 'annual',
#                 'comment': 'mass losses from calving, subaerial frontal melting, sublimation above the '
#                 'waterline and subaqueous frontal melting below the waterline'},
#         'glac_mass_annual': {
#                 'long_name': 'glacier mass',
#                 'units': 'kg',
#                 'temporal_resolution': 'annual',
#                 'comment': 'mass of ice based on area and ice thickness at start of the year'},
#         'glac_massbalclim_annual': {
#                 'long_name': 'glacier climatic mass balance',
#                 'units': 'm3',
#                 'temporal_resolution': 'annual',
#                 'comment': 'climatic mass balance is the sum of the accumulation and ablation, excluding frontal ablation'},
#         'glac_massbaltotal_annual': {
#                 'long_name': 'glacier total mass balance',
#                 'units': 'm3',
#                 'temporal_resolution': 'annual',
#                 'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
#         }
       
#     # Add variables to empty dataset and merge together
#     count_vn = 0
#     encoding = {}
#     for vn in output_coords_dict.keys():
#         count_vn += 1
#         empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
#         output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
#                                coords=output_coords_dict[vn])
#         # Merge datasets of stats into one output
#         if count_vn == 1:
#             output_ds_all = output_ds
#         else:
#             output_ds_all = xr.merge((output_ds_all, output_ds))
#     noencoding_vn = ['RGIId']
#     # Add attributes
#     for vn in output_ds_all.variables:
#         try:
#             output_ds_all[vn].attrs = output_attrs_dict[vn]
#         except:
#             pass
#         # Encoding (specify _FillValue, offsets, etc.)
#         if vn not in noencoding_vn:
#             encoding[vn] = {'_FillValue': None,
#                             'zlib':True,
#                             'complevel':9
#                             }
#     output_ds_all['RGIId'].values = np.array([glacier_rgi_table.loc['RGIId']])
#     output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
#     output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
#     output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
#     output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
#     output_ds_all['Area'].values = np.array([glacier_rgi_table.Area * 1e6])
   
#     output_ds_all.attrs = {'source': f'PyGEMv{pygem.__version__}',
#                        'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
#                        'history': 'Created by David Rounce (drounce@alaska.edu) on ' + pygem_prms.model_run_date,
#                        'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
       
#     return output_ds_all, encoding


def create_xrdataset_binned_stats(glacier_rgi_table, dates_table, surface_h_initial, 
                                  output_glac_bin_mass_annual, output_glac_bin_icethickness_annual, 
                                  output_glac_bin_massbalclim_monthly, output_glac_bin_massbalclim_annual, 
                                  output_glac_bin_dist, option_wateryear=pygem_prms.gcm_wateryear):
    """
    Create empty xarray dataset that will be used to record binned ice thickness changes

    Parameters
    ----------
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    dates_table : pandas dataframe
        table of the dates, months, days in month, etc.

    Returns
    -------
    output_ds_all : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # Create empty datasets for each variable and merge them
    # Coordinate values
    glac_values = np.array([glacier_rgi_table.name])

    # Time attributes and values
    if option_wateryear == 'hydro':
        year_type = 'water year'
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'calendar':
        year_type = 'calendar year'
        annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
    elif option_wateryear == 'custom':
        year_type = 'custom year'

    time_values = dates_table.loc[pygem_prms.gcm_spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
    time_values = [cftime.DatetimeNoLeap(x.year, x.month, x.day) for x in time_values]

    # append additional year to year_values to account for mass and area at end of period
    year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
    year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
    bin_values = np.arange(surface_h_initial.shape[0])
    
    # Variable coordinates dictionary
    output_coords_dict = collections.OrderedDict()
    output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['CenLon'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['CenLat'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['O1Region'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['O2Region'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['Area'] = collections.OrderedDict([('glac', glac_values)])
    output_coords_dict['bin_distance'] = collections.OrderedDict([('glac', glac_values), ('bin',bin_values)])
    output_coords_dict['bin_surface_h_initial'] = collections.OrderedDict([('glac', glac_values), ('bin',bin_values)])
    output_coords_dict['bin_mass_annual'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('year', year_values)]))
    output_coords_dict['bin_thick_annual'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('year', year_values)]))
    output_coords_dict['bin_massbalclim_annual'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('year', year_values)]))
    output_coords_dict['bin_massbalclim_monthly'] = (
        collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))
    if pygem_prms.sim_iters > 1:
        output_coords_dict['bin_mass_annual_mad'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('year', year_values)]))
        output_coords_dict['bin_thick_annual_mad'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('year', year_values)]))
        output_coords_dict['bin_massbalclim_annual_mad'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('year', year_values)]))
        
    # Attributes dictionary
    output_attrs_dict = {
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'bin': {
                'long_name': 'bin index',
                'comment': 'bin index referring to the glacier elevation bin'},
        'year': {
                'long_name': 'years',
                 'year_type': year_type,
                 'comment': 'years referring to the start of each year'},
        'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.0'},
        'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.0'},
        'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.0'},
        'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.0'},
        'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.0'},
        'Area': {
                'long_name': 'glacier area',
                'units': 'm2',
                'comment': 'value from RGIv6.0'},
        'bin_distance': {
                'long_name': 'distance downglacier',
                'units': 'm',
                'comment': 'horizontal distance calculated from top of glacier moving downglacier'},
        'bin_surface_h_initial': {
                'long_name': 'initial binned surface elevation',
                'units': 'm above sea level'},
        'bin_mass_annual': {
                'long_name': 'binned ice mass',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'binned ice mass at start of the year'},
        'bin_thick_annual': {
                'long_name': 'binned ice thickness',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'binned ice thickness at start of the year'},
        'bin_massbalclim_annual': {
                'long_name': 'binned climatic mass balance, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'climatic mass balance is computed before dynamics so can theoretically exceed ice thickness'},
        'bin_massbalclim_monthly' : {
                'long_name': 'binned monthly climatic mass balance, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'monthly climatic mass balance from the PyGEM mass balance module'},
        }
    if pygem_prms.sim_iters > 1:
        output_attrs_dict['bin_mass_annual_mad'] = {
                'long_name': 'binned ice mass median absolute deviation',
                'units': 'kg',
                'temporal_resolution': 'annual',
                'comment': 'mass of ice based on area and ice thickness at start of the year'}
        output_attrs_dict['bin_thick_annual_mad'] = {
                'long_name': 'binned ice thickness median absolute deviation',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'thickness of ice at start of the year'}
        output_attrs_dict['bin_massbalclim_annual_mad'] = {
                'long_name': 'binned climatic mass balance, in water equivalent, median absolute deviation',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': 'climatic mass balance is computed before dynamics so can theoretically exceed ice thickness'}
       
    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        count_vn += 1
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    noencoding_vn = ['RGIId']
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
       
        if vn not in noencoding_vn:
            encoding[vn] = {'_FillValue': None,
                            'zlib':True,
                            'complevel':9
                            }     
    output_ds_all['RGIId'].values = np.array([glacier_rgi_table.loc['RGIId']])
    output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
    output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
    output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
    output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
    output_ds_all['Area'].values = np.array([glacier_rgi_table.Area * 1e6])
    output_ds_all['bin_distance'].values = output_glac_bin_dist[np.newaxis,:]
    output_ds_all['bin_surface_h_initial'].values = surface_h_initial[np.newaxis,:]
    output_ds_all['bin_mass_annual'].values = (
            np.median(output_glac_bin_mass_annual, axis=2)[np.newaxis,:,:])
    output_ds_all['bin_thick_annual'].values = (
            np.median(output_glac_bin_icethickness_annual, axis=2)[np.newaxis,:,:])
    output_ds_all['bin_massbalclim_annual'].values = (
            np.median(output_glac_bin_massbalclim_annual, axis=2)[np.newaxis,:,:])
    output_ds_all['bin_massbalclim_monthly'].values = (
            np.median(output_glac_bin_massbalclim_monthly, axis=2)[np.newaxis,:,:])
    if pygem_prms.sim_iters > 1:
        output_ds_all['bin_mass_annual_mad'].values = (
            median_abs_deviation(output_glac_bin_mass_annual, axis=2)[np.newaxis,:,:])
        output_ds_all['bin_thick_annual_mad'].values = (
            median_abs_deviation(output_glac_bin_icethickness_annual, axis=2)[np.newaxis,:,:])
        output_ds_all['bin_massbalclim_annual_mad'].values = (
            median_abs_deviation(output_glac_bin_massbalclim_annual, axis=2)[np.newaxis,:,:])

    output_ds_all.attrs = {'source': f'PyGEMv{pygem.__version__}',
                       'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                       'history': 'Created by David Rounce (drounce@alaska.edu) on ' + pygem_prms.model_run_date,
                       'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
    return output_ds_all, encoding


def simu_MB_FA_single_glac (n_iter = None, gdir =None, modelprms = None, tau_value =None,debug = False, glacier_str = None,
                           gdir_ref = None, glacier_rgi_table = None, nyears =None, nyears_ref = None, fs = None, glen_a_multiplier = None,
                           fls = None,reg_str = None, gcm_name= None, scenario = None,args = None,count_exceed_boundary_errors =0):
    """
    Run the mass balance and flowline model for a single glacier.

    Parameters
    ----------
    n_iter : int
        iteration number
    gdir : class
        GlacierDirectory
    modelprms : dict
        dictionary of model parameters
    tau_value : list
        list of tau values for each iteration
    debug : bool
        option to turn on debug mode
    glacier_str : str
        glacier string
    gdir_ref : class
        GlacierDirectory
    glacier_rgi_table : pd.DataFrame
        table of glacier RGI information
    nyears : int
        number of years to run
    nyears_ref : int
        number of years to run for the reference run
    fs : list
        flowline objects
    glen_a_multiplier : float
        factor to increase the Glen's A parameter
    fls : list
        flowline objects
    reg_str : str
        region string
    gcm_name : str
        name of the GCM
    scenario : str
        name of the climate scenario
    args : dict
        dictionary of arguments
    count_exceed_boundary_errors : int
        count of exceed boundary errors
    
    """

    if debug:
        print('n_iter in loop through model parameters:', n_iter)
   
    if not tau_value is None:
        tau = tau_value
        cfg.PARAMS['calving_k'] = tau   #TODO Add the tau to the cfg.PARAMS, at the moment use the same symbol as the calving_k in the flowlinemodel
        cfg.PARAMS['inversion_calving_k'] = tau
    
    # successful_run used to continue runs when catching specific errors
    successful_run = True
    
    # set the suffix of each iteration
    file_suffix = '_'+str(n_iter)

    if debug:
        print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                ' tbias: ' + str(np.round(modelprms['tbias'],2)) +
                'tau :' +str(tau))

    #%%
    # ----- ICE THICKNESS INVERSION using OGGM -----
    if pygem_prms.option_dynamics is not None:
        # Apply inversion_filter on mass balance with debris to avoid negative flux
        if pygem_prms.include_debris:
            inversion_filter = True
        else:
            inversion_filter = False
                
        # Perform inversion based on PyGEM MB using reference directory
        mbmod_inv = PyGEMMassBalance(gdir_ref, modelprms, glacier_rgi_table,
                                        hindcast=pygem_prms.hindcast,
                                        debug=pygem_prms.debug_mb,
                                        debug_refreeze=pygem_prms.debug_refreeze,
                                        fls=fls, option_areaconstant=True,
                                        inversion_filter=inversion_filter)
        # print("gdir.is_tidewater is:",gdir.is_tidewater)
        # print("pygem_prms.include_calving is:",pygem_prms.include_calving)
        gdir.is_tidewater = True # TODO: this is a temporary fix to make sure the calving works, because the info in RGI60 is not correct for some tidewater glaciers
        # Non-tidewater glaciers
        if not gdir.is_tidewater or not pygem_prms.include_calving:
            # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
            apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears_ref),filesuffix=file_suffix)
            tasks.prepare_for_inversion(gdir,filesuffix=file_suffix)
            tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,filesuffix=file_suffix)

        # Tidewater glaciers
        else:
            cfg.PARAMS['use_kcalving_for_inversion'] = True
            cfg.PARAMS['use_kcalving_for_run'] = True
            print("The find_inversion_calving_from_any_mb start")
            print("⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅")
            out_calving = find_inversion_calving_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears_ref),
                                                                glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                                                calving_law_inv = None,modelprms = modelprms, 
                                                                glacier_rgi_table = glacier_rgi_table,
                                                                hindcast=pygem_prms.hindcast,debug=pygem_prms.debug_mb,
                                                                debug_refreeze=pygem_prms.debug_refreeze,option_areaconstant=True,
                                                                inversion_filter=inversion_filter,filesuffix= file_suffix)
            print("⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅")
            print("The find_inversion_calving_from_any_mb end")
            print("the out claving is:",out_calving)
        # ----- INDENTED TO BE JUST WITH DYNAMICS -----
        tasks.init_present_time_glacier(gdir,filesuffix = file_suffix) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines',filesuffix = file_suffix)  # add debris enhancement factors to flowlines

        try:
            nfls = gdir.read_pickle('model_flowlines', filesuffix = file_suffix)
        except FileNotFoundError as e:
            if 'model_flowlines.pkl' in str(e):
                tasks.compute_downstream_line(gdir)
                tasks.compute_downstream_bedshape(gdir)
                tasks.init_present_time_glacier(gdir,filesuffix = file_suffix) # adds bins below
                nfls = gdir.read_pickle('model_flowlines',filesuffix = file_suffix)
            else:
                raise

        # Water Level
        # Check that water level is within given bounds
        cls = gdir.read_pickle('inversion_input',filesuffix = file_suffix)[-1]
        th = cls['hgt'][-1]
        vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
        #water_level = utils.clip_scalar(0, th - vmax, th - vmin)
        #TODO need to check again, here is just to make sure the water level agree with the one used in the inversion 
        water_level = out_calving ['calving_water_level']
    # No ice dynamics options
    else:
        nfls = fls
        
    # Record initial surface h for overdeepening calculations
    surface_h_initial = nfls[0].surface_h
    print("the initial surface_h is :",surface_h_initial)
    # ------ MODEL WITH EVOLVING AREA ------
    # Mass balance model
    mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                hindcast=pygem_prms.hindcast,
                                debug=pygem_prms.debug_mb,
                                debug_refreeze=pygem_prms.debug_refreeze,
                                fls=nfls, option_areaconstant=False)

    # Glacier dynamics model
    if pygem_prms.option_dynamics == 'OGGM':
        print("------------------ after the thickness inversion with calving run the dynamics ------------------")
        if debug:
            print('OGGM GLACIER DYNAMICS!')
            
        # new numerical scheme is SemiImplicitModel() but doesn't have frontal ablation yet
        # FluxBasedModel is old numerical scheme but includes frontal ablation
        try:
            ev_model = CalvingFluxBasedModelJanRt(nfls, y0=0, mb_model=mbmod,
                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                    is_tidewater=gdir.is_tidewater,
                                    water_level=water_level,mb_elev_feedback=mb_elev_feedback
                                    )
            print("evmodel assigned")
        except:
            print("calving law failed")
        # if debug:
        #     graphics.plot_modeloutput_section(ev_model)
        #     plt.show()
        try:                   
            if oggm_version > 1.301:
                print("oggm ver >1.3")
                try:
                    # add the condition for different situation,1. do_fl_diag = True 2. do_fl_diag = False
                    do_fl_diag = cfg.PARAMS['store_fl_diagnostics']
                    if do_fl_diag:
                        fl_diag_path = gdir.get_filepath('fl_diagnostics',delete=True,filesuffix = file_suffix)
                        diag, fl_diag_dss = ev_model.run_until_and_store(nyears,store_monthly_step= store_monthly_step,fl_diag_path=fl_diag_path)
                        # print('diag is :',diag)
                        #pdb.set_trace()
                    else:
                        diag = ev_model.run_until_and_store(nyears,store_monthly_step= store_monthly_step)
                        # print('diag is :',diag)
                except:
                    print("oggm ver >1.3, run_until_and store failed")
                    print(traceback.format_exc())
            else:
                _, diag = ev_model.run_until_and_store(nyears)
                print("oggm run law failed")
            # print("the volume_3 in the diag is :",diag.volume_m3)
            # print("area_m2 in diag is :",diag.area_m2)
            ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
            ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
            #ev_model.mb_model.glac_wide_length_annual[-1] = diag.length_m[-1]
            # Record frontal ablation for tidewater glaciers and update total mass balance
            #pdb.set_trace()
            if gdir.is_tidewater:
                # Glacier-wide frontal ablation (m3 w.e.)
                # - note: diag.calving_m3 is cumulative calving
                try:
                    if debug:
                        print('\n\ndiag.calving_m3:', diag.calving_m3.values)
                        print('calving_m3_since_y0:', ev_model.calving_m3_since_y0)
                    calving_m3_month = ((diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1]) * 
                                        pygem_prms.density_ice / pygem_prms.density_water)
                    
                    #length_change_m_monthly = (diag.length_m.values[1:] - diag.length_m.values[0:-1])
                    #TODO check the length change rate should be [0:-1] or [1:], keep consistent with dLdt monthly in run_calibration_FA_Rt_New.py
                    length_change_m_monthly = diag.length_change_rate_myr.values[0:-1]
                    if Dynamic_step_Monthly:
                        calving_m3_annual = calving_m3_month.reshape(-1,12).sum(1)
                        length_change_m_annual = np.nanmean(length_change_m_monthly.reshape(-1, 12), axis=1)
                    else:
                        calving_m3_annual = calving_m3_month
                        length_change_m_annual = length_change_m_monthly
                    #pdb.set_trace()
                    # print("calving_m3_annual is:",calving_m3_annual)
                    print("the frontalablation is updated totally :",calving_m3_month.shape[0])
                    for n in np.arange(calving_m3_month.shape[0]):
                        # update monthly or annual
                        if Dynamic_step_Monthly:
                            ev_model.mb_model.glac_wide_frontalablation[n] = calving_m3_month[n]
                            ev_model.mb_model.glac_length_change[n] = length_change_m_annual[n]
                        else:
                            ev_model.mb_model.glac_wide_frontalablation[12*n+11] = calving_m3_annual[n]
                            ev_model.mb_model.glac_length_change[12*n+11] = length_change_m_annual[n]
                    # Glacier-wide total mass balance (m3 w.e.)
                    ev_model.mb_model.glac_wide_massbaltotal = (
                            ev_model.mb_model.glac_wide_massbaltotal  - ev_model.mb_model.glac_wide_frontalablation)
                    #pdb.set_trace()
                    if debug:
                        print("nyears is :",nyears)
                        print('avg calving_m3:', calving_m3_annual.sum() / nyears)
                        print('avg frontal ablation [Gta] (modeled FA):', 
                            np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                        print('avg frontal ablation [Gta] (calving m3 since 0):', 
                            np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                except:
                    print("calving_m3_annual update failed")
                    print(traceback.format_exc())
            print("OGGM dynamic runs successfully")
            
        except RuntimeError as e:
            if 'Glacier exceeds domain boundaries' in repr(e):
                count_exceed_boundary_errors += 1
                successful_run = False
                
                # LOG FAILURE
                fail_domain_fp = (pygem_prms.output_sim_fp + 'fail-exceed_domain/' + reg_str + '/' 
                                    + gcm_name + '/')
                if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                    fail_domain_fp += scenario + '/'
                if not os.path.exists(fail_domain_fp):
                    os.makedirs(fail_domain_fp, exist_ok=True)
                txt_fn_fail = glacier_str + "-sim_failed.txt"
                with open(fail_domain_fp + txt_fn_fail, "w") as text_file:
                    text_file.write(glacier_str + ' failed to complete ' + 
                                    str(count_exceed_boundary_errors) + ' simulations')
            elif gdir.is_tidewater:
                if debug:
                    print('OGGM dynamics failed, using mass redistribution curves (RuntimeError)')
                print("OGGM DYNAMICS FAILED, try the MassRedistributionCurves")
                # Mass redistribution curves glacier dynamics model
        except:
            if gdir.is_tidewater:
                if debug:
                    print('OGGM dynamics failed, using mass redistribution curves')
                    print(traceback.format_exc())

            else:
                raise

    # Mass redistribution model                  
    elif pygem_prms.option_dynamics == 'MassRedistributionCurves':
        if debug:
            print('MASS REDISTRIBUTION CURVES!')         
        
    elif pygem_prms.option_dynamics is None:
        # Mass balance model
        ev_model = None
        diag = xr.Dataset()
        mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                    hindcast=pygem_prms.hindcast,
                                    debug=pygem_prms.debug_mb,
                                    debug_refreeze=pygem_prms.debug_refreeze,
                                    fls=fls, option_areaconstant=True)
        # ----- MODEL RUN WITH CONSTANT GLACIER AREA -----
        years = np.arange(args.gcm_startyear, args.gcm_endyear + 1)
        mb_all = []
        for year in years - years[0]:
            mb_annual = mbmod.get_annual_mb(nfls[0].surface_h, fls=nfls, fl_id=0, year=year,
                                            debug=True)
            mb_mwea = (mb_annual * 365 * 24 * 3600 * pygem_prms.density_ice /
                        pygem_prms.density_water)
            glac_wide_mb_mwea = ((mb_mwea * mbmod.glacier_area_initial).sum() /
                                    mbmod.glacier_area_initial.sum())
            mb_all.append(glac_wide_mb_mwea)
        mbmod.glac_wide_area_annual[-1] = mbmod.glac_wide_area_annual[0]
        #mbmod.glac_wide_length_annual[-1] = mbmod.glac_wide_length_annual[0]                        
        mbmod.glac_wide_volume_annual[-1] = mbmod.glac_wide_volume_annual[0]
        diag['area_m2'] = mbmod.glac_wide_area_annual
        #diag['length_m'] = mbmod.glac_wide_length_annual
        diag['volume_m3'] = mbmod.glac_wide_volume_annual
        diag['volume_bsl_m3'] = 0
        
        if debug:
            print('iter:', n_iter, 'massbal (mean, std):', np.round(np.mean(mb_all),3), np.round(np.std(mb_all),3),
                    'massbal (med):', np.round(np.median(mb_all),3))
        
#                            mb_em_mwea = run_emulator_mb(modelprms)
#                            print('  emulator mb:', np.round(mb_em_mwea,3))
#                            mb_em_sims.append(mb_em_mwea)
    
    print("successful_run is :",successful_run)

    # return variables
    return (n_iter,successful_run, ev_model, diag, mbmod, surface_h_initial,length_change_m_annual,calving_m3_annual,nfls,count_exceed_boundary_errors)


def process_for_parallel(n_iter= None, model_function =None,modelprms_all =None,tau_values = None,**kwargs):
    """
    Wrapper for parallel processing of parameter samples
    Parameters
    ----------
    n_iter : int
        iteration number
    model_function : func
        function to run, here is simu_MB_FA_single_glac 
    modelprms_MB_FA: dict
        dictionary of model parameters    
    tau_values : array
        array of tau values
    kwargs : dict
        dictionary of keyword arguments of the function
    Returns
    -------
    out_dict: dict
        dictionary of output variables
    """
    # Extract parameter values for this iteration
    modelprms_iter = {key: value[n_iter] for key, value in modelprms_all.items()}
    tau_values_iter = tau_values[n_iter]

    # Run the model function with iteration-specific parameters
    out = model_function(modelprms=modelprms_iter, tau_value=tau_values_iter, n_iter=n_iter, **kwargs)

    return out

    
def simu_MB_FA(list_packed_vars,num_cores = 1,model_function = simu_MB_FA_single_glac,do_DA_simulation = True,**kwargs):
    """
    Model simulation (MB+FA)
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    num_cores : int
        number of cores to use for parallel processing, the default is 1

    model_function : func
        function to run, the default is simu_MB_FA_single_glac

    do_DA_simulation : bool
        option to run the data assimilation simulation, the default is True

    kwargs : Any
        function arguments

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    parser = getparser()
    args = parser.parse_args()
    #count = list_packed_vars[0]
    glac_no = list_packed_vars[0]
    gcm_name = list_packed_vars[1]
    realization = list_packed_vars[2]
    scenario = list_packed_vars[3]
    # if (gcm_name != pygem_prms.ref_gcm_name) and (args.scenario is None):
    #     scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
    # elif not args.scenario is None:
    #     scenario = args.scenario
    debug = args.debug
    # if debug:
    #     if 'scenario' in locals():
    #         print(scenario)
    if args.debug_spc:
        debug_spc = True
    else:
        debug_spc = False
    debug = True
    debug_spc = True
    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    
    
    # ===== TIME PERIOD =====
    # Reference Calibration Period
    #  adjust end year in event that reference and GCM don't align
    if pygem_prms.ref_endyear <= args.gcm_endyear:
        ref_endyear = pygem_prms.ref_endyear
    else:
        ref_endyear = args.gcm_endyear
    dates_table_ref = modelsetup.datesmodelrun(startyear=pygem_prms.ref_startyear, endyear=ref_endyear,
                                               spinupyears=pygem_prms.ref_spinupyears,
                                               option_wateryear=pygem_prms.ref_wateryear)
    # Reference Bias Adjustment Period
    dates_table_ref_bc = modelsetup.datesmodelrun(startyear=args.gcm_bc_startyear, endyear=ref_endyear,
                                                  spinupyears=pygem_prms.ref_spinupyears,
                                                  option_wateryear=pygem_prms.ref_wateryear)
    
    if debug:
        print('ref years:', pygem_prms.ref_startyear, ref_endyear)
        print('ref bc years:', args.gcm_bc_startyear, ref_endyear)
        
    # GCM Full Period (includes bias correction and simulation)
    if pygem_prms.ref_startyear <= args.gcm_startyear:
        gcm_startyear = pygem_prms.ref_startyear
    else:
        gcm_startyear = args.gcm_startyear
        
    dates_table_full = modelsetup.datesmodelrun(
            startyear=gcm_startyear, endyear=args.gcm_endyear, spinupyears=pygem_prms.gcm_spinupyears,
            option_wateryear=pygem_prms.gcm_wateryear)
    
    # GCM Simulation Period
    if args.gcm_startyear > gcm_startyear:
        dates_table = modelsetup.datesmodelrun(
                startyear=args.gcm_startyear, endyear=args.gcm_endyear, spinupyears=pygem_prms.gcm_spinupyears,
                option_wateryear=pygem_prms.gcm_wateryear)
    else:
        dates_table = dates_table_full
    
    
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    if gcm_name in ['ERA5', 'ERA-Interim', 'COAWST']:
        gcm = class_climate.GCM(name=gcm_name)
        ref_gcm = gcm
        dates_table_ref = dates_table_full
    else:
        # GCM object
        if realization is None:
            gcm = class_climate.GCM(name=gcm_name, scenario=scenario)
        else:
            gcm = class_climate.GCM(name=gcm_name, scenario=scenario, realization=realization)
        # Reference GCM
        ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    
    # ----- Select Temperature and Precipitation Data -----
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                 dates_table_full)
    ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn,
                                                                     main_glac_rgi, dates_table_ref_bc)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                 dates_table_full)
    ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn,
                                                                     main_glac_rgi, dates_table_ref_bc)
    # Elevation [m asl]
    try:
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    except:
        gcm_elev = None
    ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
    
    # ----- Temperature and Precipitation Bias Adjustments -----
    # No adjustments
    if pygem_prms.option_bias_adjustment == 0 or gcm_name == pygem_prms.ref_gcm_name:
        if pygem_prms.gcm_wateryear == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        sim_idx_start = dates_table_full[dates_cn].to_list().index(pygem_prms.gcm_startyear)
        gcm_elev_adj = gcm_elev
        gcm_temp_adj = gcm_temp[:,sim_idx_start:]
        gcm_prec_adj = gcm_prec[:,sim_idx_start:]
    # Bias correct based on reference climate data
    else:
        # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
        if pygem_prms.option_bias_adjustment == 1:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp,
                                                                        dates_table_ref, dates_table_full,
                                                                        ref_spinupyears=pygem_prms.ref_spinupyears,
                                                                        gcm_spinupyears=pygem_prms.gcm_spinupyears)
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec,
                                                                      dates_table_ref, dates_table_full,
                                                                      ref_spinupyears=pygem_prms.ref_spinupyears,
                                                                      gcm_spinupyears=pygem_prms.gcm_spinupyears)
        # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
        elif pygem_prms.option_bias_adjustment == 2:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp,
                                                                        dates_table_ref, dates_table_full,
                                                                        ref_spinupyears=pygem_prms.ref_spinupyears,
                                                                        gcm_spinupyears=pygem_prms.gcm_spinupyears)
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec,
                                                                        dates_table_ref, dates_table_full,
                                                                        ref_spinupyears=pygem_prms.ref_spinupyears,
                                                                        gcm_spinupyears=pygem_prms.gcm_spinupyears)
        # OPTION 3: Adjust temp and prec using quantile delta mapping, Cannon et al. (2015)
        elif pygem_prms.option_bias_adjustment == 3:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_QDM(ref_temp, ref_elev, gcm_temp,
                                                                      dates_table_ref, dates_table_full,
                                                                      ref_spinupyears=pygem_prms.ref_spinupyears,
                                                                      gcm_spinupyears=pygem_prms.gcm_spinupyears)


            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_QDM(ref_prec, ref_elev, gcm_prec,
                                                                      dates_table_ref, dates_table_full,
                                                                      ref_spinupyears=pygem_prms.ref_spinupyears,
                                                                      gcm_spinupyears=pygem_prms.gcm_spinupyears)
    
    # assert that the gcm_elev_adj is not None
    assert gcm_elev_adj is not None, 'No GCM elevation data'

    # ----- Update Reference Period to be consistent with calibration period -----
    if pygem_prms.ref_startyear != args.gcm_bc_startyear:
        if pygem_prms.gcm_wateryear == 'hydro':
            dates_cn = 'wateryear'
        else:
            dates_cn = 'year'
        ref_idx_start = dates_table_ref_bc[dates_cn].to_list().index(pygem_prms.ref_startyear)
        ref_temp = ref_temp[:,ref_idx_start:]
        ref_prec = ref_prec[:,ref_idx_start:]
    
    # ----- Other Climate Datasets (Air temperature variability [degC] and Lapse rate [K m-1])
    # Air temperature variability [degC]
    if pygem_prms.option_ablation != 2:
        gcm_tempstd = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
        ref_tempstd = np.zeros((main_glac_rgi.shape[0],dates_table_ref.shape[0]))
    elif pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                        main_glac_rgi, dates_table)
        ref_tempstd = gcm_tempstd
    elif pygem_prms.option_ablation == 2 and pygem_prms.ref_gcm_name in ['ERA5']:
        # Compute temp std based on reference climate data
        ref_tempstd, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.tempstd_fn, ref_gcm.tempstd_vn,
                                                                            main_glac_rgi, dates_table_ref)
        # Monthly average from reference climate data
        gcm_tempstd = gcmbiasadj.monthly_avg_array_rolled(ref_tempstd, dates_table_ref, dates_table_full)
    else:
        gcm_tempstd = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
        ref_tempstd = np.zeros((main_glac_rgi.shape[0],dates_table_ref.shape[0]))

    # Lapse rate
    if gcm_name in ['ERA-Interim', 'ERA5']:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
        ref_lr = gcm_lr
    else:
        # Compute lapse rates based on reference climate data
        ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi,
                                                                        dates_table_ref)
        # Monthly average from reference climate data
        gcm_lr = gcmbiasadj.monthly_avg_array_rolled(ref_lr, dates_table_ref, dates_table_full)
        
    
    # ===== RUN MASS BALANCE =====
    print("==================================== Run Mass BALANCE START ====================================")
   
    # Number of years (for OGGM's run_until_and_store)
    if pygem_prms.timestep == 'monthly':
        nyears = int(dates_table.shape[0]/12)
        nyears_ref = int(dates_table_ref.shape[0]/12)
        print("nyears is :",nyears)
        print("nyears_ref :",nyears_ref)
    else:
        assert True==False, 'Adjust nyears for non-monthly timestep'

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :].copy()
        # TODO It's temporal setting, here we hardcode the termtype as 1, because the info in RGI60 is not correct for some tidewater glaciers
        glacier_rgi_table['TermType'] = 1
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        reg_str = str(glacier_rgi_table.O1Region).zfill(2)
        rgiid = main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId']

        # Log failure
        fail_fp = pygem_prms.output_sim_fp + 'failed/' + reg_str + '/' + gcm_name + '/'
        if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
            fail_fp += scenario + '/'
        if not os.path.exists(fail_fp):
            os.makedirs(fail_fp, exist_ok=True)
        # Log the failure message to sim_failed.txt, appending if it already exists
        txt_fn_fail = glacier_str + "-sim_failed.txt"
        txt_fp_fail = os.path.join(fail_fp, txt_fn_fail)
        # Log the traceback information to sim_traceback.txt, appending if it already exists
        traceback_fn = glacier_str + "-sim_traceback.txt"
        traceback_fp = os.path.join(fail_fp, traceback_fn)
        # Log the warning message to sim_warning.txt, appending if it already exists
        warning_fn = glacier_str + "-sim_warning.txt"
        warning_fp = os.path.join(fail_fp, warning_fn)

        print("====================================")
        print("-------------------------- start MB Running for the",glac,"glacier, rgiid is :",rgiid,"--------------------------")
        # if do_DA_simulation:
        #     k_str=glacier_str+"_"+f"{index_pariticles:.0f}"
        # else:
        #     k_str = ''

        try:
        # for batman in [0]:

            # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                gdir = single_flowline_glacier_directory(glacier_str, logging_level=pygem_prms.logging_level)
                gdir.is_tidewater = False
                tau_values = None
            else:
                gdir = single_flowline_glacier_directory_with_calving(glacier_str, logging_level=pygem_prms.logging_level)
                gdir.is_tidewater = True
                cfg.PARAMS['use_kcalving_for_inversion'] = True
                cfg.PARAMS['use_kcalving_for_run'] = True

            # Flowlines
            fls = gdir.read_pickle('inversion_flowlines')
    
            # Reference gdir for ice thickness inversion
            gdir_ref = copy.deepcopy(gdir)
            gdir_ref.historical_climate = {'elev': ref_elev[glac],
                                        'temp': ref_temp[glac,:],
                                        'tempstd': ref_tempstd[glac,:],
                                        'prec': ref_prec[glac,:],
                                        'lr': ref_lr[glac,:]}
            gdir_ref.dates_table = dates_table_ref

            # Add climate data to glacier directory
            if pygem_prms.hindcast == True:
                gcm_temp_adj = gcm_temp_adj[::-1]
                gcm_tempstd = gcm_tempstd[::-1]
                gcm_prec_adj= gcm_prec_adj[::-1]
                gcm_lr = gcm_lr[::-1]
                
            gdir.historical_climate = {'elev': gcm_elev_adj[glac],
                                        'temp': gcm_temp_adj[glac,:],
                                        'tempstd': gcm_tempstd[glac,:],
                                        'prec': gcm_prec_adj[glac,:],
                                        'lr': gcm_lr[glac,:]}
            gdir.dates_table = dates_table
            
            glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
            if (fls is not None) and (glacier_area_km2.sum() > 0):
                
                # Load model parameters
                if pygem_prms.use_calibrated_modelparams: #TODO  set it as  the input args


                    # Load calibrated parameters
                    #output_fp = args.output_fp #TODO check the status of output fp, now it's global
                    modelprms_fp = args.modelprms_fp
                    if not modelprms_fp: 
                        # # The full posterior parameter set                   
                        # modelprms_fn = f'calibration_poster_Params_{rgiid}.json' 
                        # modelprms_fp = os.path.join(output_fp, 'parameter','Poster')
                        # modelprms_fullfn = os.path.join(modelprms_fp, modelprms_fn)
                        # The unique posterior parameter set
                        modelprms_fn = f'calibration_poster_Params_unique_{rgiid}.json' 
                        modelprms_fp = os.path.join(output_fp_cali, 'parameter', reg_str, glacier_str, 'Poster', 'Unique')
                        modelprms_fullfn = os.path.join(modelprms_fp, modelprms_fn)
                    print("modelprms_fullfn :",modelprms_fullfn)    
                    assert os.path.exists(modelprms_fullfn), 'Calibrated parameters do not exist.'
                    
                    with open(modelprms_fullfn, 'r') as f:
                        modelprms_dict = json.load(f)
    
                    modelprms_all = modelprms_dict
                    # add the ddfice in the modelprms_all
                    modelprms_all['ddfice'] = np.array(modelprms_all['ddfsnow'])/pygem_prms.ddfsnow_iceratio
                    modelprms_all['tsnow_threshold'] = pygem_prms.tsnow_threshold
                    modelprms_all['precgrad'] = pygem_prms.precgrad
                    # PBS/AMIS needs model parameters to be selected
                    if pygem_prms.option_calibration == 'PBS':
                        sim_iters = len(modelprms_all['tbias'])
                        if sim_iters == 1:
                            modelprms_all = {'kp': [np.median(modelprms_all['kp'])],
                                              'tbias': [np.median(modelprms_all['tbias'])],
                                              'ddfsnow': [np.median(modelprms_all['ddfsnow'])],
                                              'ddfice': [np.median(modelprms_all['ddfice'])],
                                              'tau': [np.median(modelprms_all['tau'])],
                                              'tsnow_threshold':  pygem_prms.tsnow_threshold ,
                                              'precgrad': pygem_prms.precgrad
                                              }
                            tau_values = np.median(modelprms_all['tau'])
                        else:
                            # Select every kth iteration to use for the ensemble
                            # mcmc_sample_no = len(modelprms_all['kp']['chain_0'])
                            # mp_spacing = int((mcmc_sample_no - pygem_prms.sim_burn) / sim_iters)
                            # mp_idx_start = np.arange(pygem_prms.sim_burn, pygem_prms.sim_burn + mp_spacing)
                            # np.random.shuffle(mp_idx_start)
                            # mp_idx_start = mp_idx_start[0]
                            # mp_idx_all = np.arange(mp_idx_start, mcmc_sample_no, mp_spacing)
                            modelprms_all = {
                                    'kp': modelprms_all['kp'],
                                    'tbias': modelprms_all['tbias'],
                                    'ddfsnow': modelprms_all['ddfsnow'],
                                    'ddfice': modelprms_all['ddfice'],
                                    'tau': modelprms_all['tau'],
                                    'tsnow_threshold': [modelprms_all['tsnow_threshold']] * sim_iters,
                                    'precgrad': [modelprms_all['precgrad']] * sim_iters}
                            tau_values = np.array(modelprms_all['tau'])
                    else:
                        sim_iters = pygem_prms.sim_iters
                else:
                    modelprms_all = {'kp': [pygem_prms.kp],
                                      'tbias': [pygem_prms.tbias],
                                      'ddfsnow': [pygem_prms.ddfsnow],
                                      'ddfice': [pygem_prms.ddfice],
                                      'tsnow_threshold': [pygem_prms.tsnow_threshold],
                                      'precgrad': [pygem_prms.precgrad]}
                    tau = np.zeros(sim_iters) + pygem_prms.calving_k #TODO add pygem_prms.tau
                    tau_values = tau
                    
                if debug and gdir.is_tidewater:
                    print('tau_values:', tau_values)
                    

                # Load OGGM glacier dynamics parameters (if necessary)
                if pygem_prms.option_dynamics in ['OGGM', 'MassRedistributionCurves']:

                    # CFL number (may use different values for calving to prevent errors)
                    if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                        cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number
                    else:
                        cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number_calving

                    
                    if debug:
                        print('cfl number:', cfg.PARAMS['cfl_number'])
                        
                    if pygem_prms.use_reg_glena:
                        glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)                    
                        glena_O1regions = [int(x) for x in glena_df.O1Region.values]
                        assert glacier_rgi_table.O1Region in glena_O1regions, glacier_str + ' O1 region not in glena_df'
                        glena_idx = np.where(glena_O1regions == glacier_rgi_table.O1Region)[0][0]
                        glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                        fs = glena_df.loc[glena_idx,'fs']
                        fs = cfg.PARAMS['fs']
                    else:
                        fs = pygem_prms.fs
                        glen_a_multiplier = pygem_prms.glen_a_multiplier
    
                # Time attributes and values
                if pygem_prms.gcm_wateryear == 'hydro':
                    annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
                else:
                    annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
                # append additional year to year_values to account for mass and area at end of period
                year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
                print("year_values is:",year_values)
                year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
                print("year_value now is :",year_values)
                output_glac_temp_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_prec_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_acc_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_refreeze_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_melt_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_frontalablation_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_length_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_length_change_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan                
                output_glac_massbaltotal_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_massbalclim_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_runoff_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_snowline_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_area_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_length_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_length_change_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_frontalablation_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_massbaltotal_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_massbalclim_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_mass_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_mass_bsl_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_glac_mass_change_ignored_annual = np.zeros((year_values.shape[0], sim_iters))
                output_glac_ELA_annual = np.zeros((year_values.shape[0], sim_iters)) * np.nan
                output_offglac_prec_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_offglac_refreeze_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_offglac_melt_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_offglac_snowpack_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_offglac_runoff_monthly = np.zeros((dates_table.shape[0], sim_iters)) * np.nan
                output_glac_bin_icethickness_annual = None
               
                # Loop through model parameters, parallel processing
                count_exceed_boundary_errors = 0
                mb_em_sims = []

                num_iterations = sim_iters
                process_func = partial(process_for_parallel, model_function = model_function, gdir = gdir, modelprms_all = modelprms_all, tau_values = tau_values,
                                        glacier_str = glacier_str,gdir_ref = gdir_ref, glacier_rgi_table = glacier_rgi_table, nyears =nyears, nyears_ref = nyears_ref,
                                        fs = fs, glen_a_multiplier = glen_a_multiplier, fls = fls,reg_str = reg_str, gcm_name= gcm_name, scenario = scenario,args = args,
                                        count_exceed_boundary_errors = count_exceed_boundary_errors, **kwargs)
                if num_iterations > 1 and num_cores > 1:
                    with Pool(num_cores) as pool:
                            output_parallel = pool.map(process_func, range(num_iterations))
                    # Sequential processing
                else:
                    output_parallel = [process_func(n_iter) for n_iter in range(num_iterations)]  
                #         output_parallel = [
                #             model_function(n_iter=n_iter, modelprms={key: value[n_iter] for key, value in modelprms_all.items()}, 
                #                         tau_value=tau_values[n_iter], **kwargs)
                #             for n_iter in range(num_iterations)]            
                # else:
                #     # Single iteration case
                #     output_parallel = [model_function(n_iter=0, modelprms={key: value[0] for key, value in modelprms_all.items()}, 
                #                                     tau_value=tau_values[0], **kwargs)]             
                # Extract results from parallel processing

                # Extract results from parallel processing
                # Sort results by iteration index (first element of each tuple) to preserve order
                output_parallel.sort(key=lambda x: x[0])
                # Extract each variable separately
                successful_runs = [result[1] for result in output_parallel]   # List of success flags
                ev_models = np.array([result[2] for result in output_parallel])  # Evaluation models
                diags = [result[3] for result in output_parallel]  # Diagnostic info
                mbmods = np.array([result[4] for result in output_parallel])  # Glacier-wide temp
                surface_h_initials = np.array([result[5] for result in output_parallel])  # Surface heights
                length_change_m_annuals = np.array([result[6] for result in output_parallel])  # Length change
                calving_m3_annuals = np.array([result[7] for result in output_parallel])  # Calving volume
                nflss = [result[8] for result in output_parallel]  # Flowlines
                count_exceed_boundary_errors = sum([result[9] for result in output_parallel])  # Count of boundary errors

                # extract the value for each iteration
                for n_iter in range(num_iterations):
                    successful_run = successful_runs [n_iter]
                    ev_model = ev_models [n_iter]
                    diag = diags [n_iter]
                    mbmod = mbmods [n_iter]
                    surface_h_initial = surface_h_initials [n_iter]
                    length_change_m_annual = length_change_m_annuals [n_iter]
                    calving_m3_annual = calving_m3_annuals [n_iter]
                    nfls = nflss [n_iter]

                    # Record output for successful runs
                    if successful_run:

                        if not pygem_prms.option_dynamics is None:
                            #                 if debug:
                            #                     graphics.plot_modeloutput_section(ev_model)
                            # #                    graphics.plot_modeloutput_map(gdir, model=ev_model)
                            #                     plt.figure()
                            #                     diag.volume_m3.plot()
                            #                     plt.figure()
                            #                     diag.area_m2.plot()
                            #                     plt.show()

                            # Post-process data to ensure mass is conserved and update accordingly for ignored mass losses
                            #  ignored mass losses occur because mass balance model does not know ice thickness and flux divergence
                            area_initial = mbmod.glac_bin_area_annual[:, 0].sum()
                            # %% Dynamic running step
                            # if the dynamic step is monthly, the volume is monthly, need to be calculated as annual
                            # Dynamic_step_Monthly =True
                            # if Dynamic_step_Monthly :
                            #     mb_mwea_diag = ((diag.volume_m3.values[-1] - diag.volume_m3.values[0]) 
                            #                     / area_initial / nyears/12 * pygem_prms.density_ice / pygem_prms.density_water)
                            # else:
                            mb_mwea_diag = ((diag.volume_m3.values[-1] - diag.volume_m3.values[0])
                                            / area_initial / nyears * pygem_prms.density_ice / pygem_prms.density_water)
                            # %%                            
                            mb_mwea_mbmod = mbmod.glac_wide_massbaltotal.sum() / area_initial / nyears

                            # .set_trace()

                            if debug:
                                vol_change_diag = diag.volume_m3.values[-1] - diag.volume_m3.values[0]
                                print('  vol init  [Gt]:', np.round(diag.volume_m3.values[0] * 0.9 / 1e9, 5))
                                print('  vol final [Gt]:',
                                      np.round((diag.volume_m3.values[-1] - diag.volume_m3.values[-2]) * 0.9 / 1e9, 5))
                                print('  vol change[Gt] :', np.round(vol_change_diag * 0.9 / 1e9, 5))
                                print('  mb [mwea]:', np.round(mb_mwea_diag, 2))
                                print('  mb_mbmod [mwea]:', np.round(mb_mwea_mbmod, 2))

                            # print("mb_mwea_diag is :", mb_mwea_diag)
                            # print("mb_mwea_mbmod is :", mb_mwea_mbmod)
                            # print("diag is :", diag)
                            try:
                                if np.abs(mb_mwea_diag - mb_mwea_mbmod) > 1e-6:
                                    print("np.abs(mb_mwea_diag - mb_mwea_mbmod) > 1e-6")
                                    ev_model.mb_model.ensure_mass_conservation(diag,
                                                                               Dynamic_step_Monthly=Dynamic_step_Monthly)
                            except:
                                if debug:
                                    print(traceback.format_exc())

                        if debug:
                            print('Total mass balance total[Gt]:', mbmod.glac_wide_massbaltotal.sum() / 1e9)
                            print('the glacier wide frontalablation [Gt]is :',
                                  mbmod.glac_wide_frontalablation.sum() / 1e9)

                        # RECORD PARAMETERS TO DATASET
                        # print("area_m2 in diag :", diag.area_m2.values)
                        # print("length_m in diag :",diag.length_m.values)                        
                        # print("volume_bsl_m3.values :", diag.volume_bsl_m3.values)
                        try:
                            output_glac_temp_monthly[:, n_iter] = mbmod.glac_wide_temp
                            output_glac_prec_monthly[:, n_iter] = mbmod.glac_wide_prec
                            output_glac_acc_monthly[:, n_iter] = mbmod.glac_wide_acc
                            output_glac_refreeze_monthly[:, n_iter] = mbmod.glac_wide_refreeze
                            output_glac_melt_monthly[:, n_iter] = mbmod.glac_wide_melt
                            output_glac_frontalablation_monthly[:, n_iter] = mbmod.glac_wide_frontalablation
                            output_glac_massbaltotal_monthly[:, n_iter] = mbmod.glac_wide_massbaltotal
                            output_glac_massbalclim_monthly[:, n_iter] = mbmod.glac_wide_massbalclim
                            output_glac_runoff_monthly[:, n_iter] = mbmod.glac_wide_runoff
                            output_glac_snowline_monthly[:, n_iter] = mbmod.glac_wide_snowline
                            output_glac_massbaltotal_annual[:, n_iter] = np.append(np.nansum(mbmod.glac_wide_massbaltotal.reshape(-1, 12), axis=1),
                                                                                 mbmod.glac_wide_massbaltotal[-1]) 
                            output_glac_massbalclim_annual[:, n_iter] = np.append(np.nansum(mbmod.glac_wide_massbalclim.reshape(-1, 12), axis=1),
                                                                                 mbmod.glac_wide_massbalclim[-1])
                            #pdb.set_trace()
                            if Dynamic_step_Monthly:
                                # the length_m read from the flowline
                                output_glac_length_monthly[:, n_iter] = diag.length_m.values[:-1]
                                # this length change is from the SermeQ
                                output_glac_length_change_monthly[:, n_iter] = mbmod.glac_length_change
                            else:
                                # pdb.set_trace()
                                output_glac_length_change_annual[:, n_iter] = np.append(length_change_m_annual,
                                                                                        diag.length_change_rate_myr.values[
                                                                                            -1])
                                output_glac_frontalablation_annual[:, n_iter] = np.append(calving_m3_annual,
                                                                                          calving_m3_annual[-1])

                            # calving_m3_month = ((diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1]) * 
                            #                           pygem_prms.density_ice / pygem_prms.density_water)
                            # calving_m3_annual = calving_m3_month.reshape(-1,12).sum(1)

                            # output_glac_area_annual[:, n_iter] = diag.area_m2.values

                            # %% Dynamic running step
                            # if the dynamic step is monthly, the volume is monthly, need to be calculated as annual
                            # Dynamic_step_Monthly =True
                            try:
                                if Dynamic_step_Monthly:
                                    area_m2_annual = (((diag.area_m2.values[:-1]).reshape(-1, 12))[:, 0]).flatten()
                                    length_m_annual = (((diag.length_m.values[:-1]).reshape(-1, 12))[:, 0]).flatten()
                                    volume_m3_annual = (((diag.volume_m3.values[:-1]).reshape(-1, 12))[:, 0]).flatten()
                                    volume_bsl_annual = (
                                    ((diag.volume_bsl_m3.values[:-1]).reshape(-1, 12))[:, 0]).flatten()

                                    output_glac_area_annual[:, n_iter] = np.append(area_m2_annual,
                                                                                   diag.area_m2.values[-1])
                                    output_glac_length_annual[:, n_iter] = np.append(length_m_annual,
                                                                                     diag.length_m.values[-1])
                                    output_glac_mass_annual[:, n_iter] = (np.append(volume_m3_annual,
                                                                                    diag.volume_m3.values[
                                                                                        -1])) * pygem_prms.density_ice
                                    output_glac_mass_bsl_annual[:, n_iter] = (np.append(volume_bsl_annual,
                                                                                        diag.volume_bsl_m3.values[
                                                                                            -1])) * pygem_prms.density_ice
                                else:
                                    output_glac_area_annual[:, n_iter] = diag.area_m2.values
                                    output_glac_length_annual[:, n_iter] = diag.length_m.values
                                    output_glac_mass_annual[:, n_iter] = diag.volume_m3.values * pygem_prms.density_ice
                                    output_glac_mass_bsl_annual[:,
                                    n_iter] = diag.volume_bsl_m3.values * pygem_prms.density_ice
                            except:
                                if debug:
                                    print(traceback.format_exc())

                            output_glac_mass_change_ignored_annual[:-1,
                            n_iter] = mbmod.glac_wide_volume_change_ignored_annual * pygem_prms.density_ice
                            output_glac_ELA_annual[:, n_iter] = mbmod.glac_wide_ELA_annual
                            output_offglac_prec_monthly[:, n_iter] = mbmod.offglac_wide_prec
                            output_offglac_refreeze_monthly[:, n_iter] = mbmod.offglac_wide_refreeze
                            output_offglac_melt_monthly[:, n_iter] = mbmod.offglac_wide_melt
                            output_offglac_snowpack_monthly[:, n_iter] = mbmod.offglac_wide_snowpack
                            output_offglac_runoff_monthly[:, n_iter] = mbmod.offglac_wide_runoff
                            print("------------- record 1 finish ------------- ")
                        except:
                            if debug:
                                print(traceback.format_exc())

                        if output_glac_bin_icethickness_annual is None:
                            try:
                                output_glac_bin_mass_annual_sim = (mbmod.glac_bin_area_annual *
                                                                   mbmod.glac_bin_icethickness_annual *
                                                                   pygem_prms.density_ice)[:, :, np.newaxis]
                                output_glac_bin_icethickness_annual_sim = (mbmod.glac_bin_icethickness_annual)[:, :,
                                                                          np.newaxis]
                                # Update the latest thickness and volume
                                if ev_model is not None:
                                    fl_dx_meter = getattr(ev_model.fls[0], 'dx_meter', None)
                                    fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
                                    fl_section = getattr(ev_model.fls[0], 'section', None)
                                else:
                                    fl_dx_meter = getattr(nfls[0], 'dx_meter', None)
                                    fl_widths_m = getattr(nfls[0], 'widths_m', None)
                                    fl_section = getattr(nfls[0], 'section', None)
                                if fl_section is not None and fl_widths_m is not None:
                                    # thickness
                                    icethickness_t0 = np.zeros(fl_section.shape)
                                    icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[
                                        fl_widths_m > 0]
                                    output_glac_bin_icethickness_annual_sim[:, -1, 0] = icethickness_t0
                                    # mass
                                    glacier_vol_t0 = fl_widths_m * fl_dx_meter * icethickness_t0
                                    output_glac_bin_mass_annual_sim[:, -1, 0] = glacier_vol_t0 * pygem_prms.density_ice
                                output_glac_bin_mass_annual = output_glac_bin_mass_annual_sim
                                output_glac_bin_icethickness_annual = output_glac_bin_icethickness_annual_sim
                                output_glac_bin_massbalclim_annual_sim = np.zeros(
                                    mbmod.glac_bin_icethickness_annual.shape)
                                output_glac_bin_massbalclim_annual_sim[:, :-1] = mbmod.glac_bin_massbalclim_annual
                                output_glac_bin_massbalclim_annual = output_glac_bin_massbalclim_annual_sim[:, :,
                                                                     np.newaxis]
                                output_glac_bin_massbalclim_monthly_sim = np.zeros(mbmod.glac_bin_massbalclim.shape)
                                output_glac_bin_massbalclim_monthly_sim = mbmod.glac_bin_massbalclim
                                output_glac_bin_massbalclim_monthly = output_glac_bin_massbalclim_monthly_sim[:, :,
                                                                      np.newaxis]
                                print(
                                    "------------- record 2 finish ------------- (output_glac_bin_icethickness_annual is None)")
                            except:
                                print(traceback.format_exc())
                        else:
                            # print("output_glac_bin_icethickness_annual is Not None:",
                            #       output_glac_bin_icethickness_annual)
                            # Update the latest thickness and volume
                            output_glac_bin_mass_annual_sim = (mbmod.glac_bin_area_annual *
                                                               mbmod.glac_bin_icethickness_annual *
                                                               pygem_prms.density_ice)[:, :, np.newaxis]
                            output_glac_bin_icethickness_annual_sim = (mbmod.glac_bin_icethickness_annual)[:, :,
                                                                      np.newaxis]
                            if ev_model is not None:
                                fl_dx_meter = getattr(ev_model.fls[0], 'dx_meter', None)
                                fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
                                fl_section = getattr(ev_model.fls[0], 'section', None)
                            else:
                                fl_dx_meter = getattr(nfls[0], 'dx_meter', None)
                                fl_widths_m = getattr(nfls[0], 'widths_m', None)
                                fl_section = getattr(nfls[0], 'section', None)
                            if fl_section is not None and fl_widths_m is not None:
                                # thickness
                                icethickness_t0 = np.zeros(fl_section.shape)
                                icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[
                                    fl_widths_m > 0]
                                output_glac_bin_icethickness_annual_sim[:, -1, 0] = icethickness_t0
                                # mass
                                glacier_vol_t0 = fl_widths_m * fl_dx_meter * icethickness_t0
                                output_glac_bin_mass_annual_sim[:, -1, 0] = glacier_vol_t0 * pygem_prms.density_ice
                            output_glac_bin_mass_annual = np.append(output_glac_bin_mass_annual,
                                                                    output_glac_bin_mass_annual_sim, axis=2)
                            output_glac_bin_icethickness_annual = np.append(output_glac_bin_icethickness_annual,
                                                                            output_glac_bin_icethickness_annual_sim,
                                                                            axis=2)
                            output_glac_bin_massbalclim_annual_sim = np.zeros(mbmod.glac_bin_icethickness_annual.shape)
                            output_glac_bin_massbalclim_annual_sim[:, :-1] = mbmod.glac_bin_massbalclim_annual
                            output_glac_bin_massbalclim_annual = np.append(output_glac_bin_massbalclim_annual,
                                                                           output_glac_bin_massbalclim_annual_sim[:, :,
                                                                           np.newaxis],
                                                                           axis=2)


                # ===== Export Results =====
                print("count_exceed_boundary_errors is",count_exceed_boundary_errors)
                if count_exceed_boundary_errors < num_iterations:
                    
                    # ----- STATS OF ALL VARIABLES -----
                    if pygem_prms.export_essential_data:
                        try:
                            #pdb.set_trace()
                            # Create empty dataset
                            output_ds_all_stats, encoding = create_xrdataset(glacier_rgi_table, dates_table)
                            # Create empty dataset for all variables and all statistic infomations
                            output_ds_all_stats_ALL, encoding_ALL = create_xrdataset_all_statis(glacier_rgi_table, dates_table)
                            # Output statistics
                            output_glac_runoff_monthly_stats = calc_stats_array_Restruct(output_glac_runoff_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_runoff_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_area_annual_stats = calc_stats_array_Restruct(output_glac_area_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_area_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_length_annual_stats = calc_stats_array_Restruct(output_glac_length_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_length_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_length_change_annual_stats = calc_stats_array_Restruct(output_glac_length_change_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_length_change_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_frontalablation_annual_stats = calc_stats_array_Restruct(output_glac_frontalablation_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_frontalablation_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_massbaltotal_annual_stats = calc_stats_array_Restruct(output_glac_massbaltotal_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_massbaltotal_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_massbalclim_annual_stats = calc_stats_array_Restruct(output_glac_massbalclim_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_massbalclim_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_mass_annual_stats = calc_stats_array_Restruct(output_glac_mass_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_mass_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_mass_bsl_annual_stats = calc_stats_array_Restruct(output_glac_mass_bsl_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_mass_bsl_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_glac_ELA_annual_stats = calc_stats_array_Restruct(output_glac_ELA_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_ELA_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            output_offglac_runoff_monthly_stats = calc_stats_array_Restruct(output_offglac_runoff_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'offglac_runoff_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            if pygem_prms.export_extra_vars:
                                output_glac_temp_monthly_stats = calc_stats_array_Restruct(output_glac_temp_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_temp_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_prec_monthly_stats = calc_stats_array_Restruct(output_glac_prec_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_prec_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_acc_monthly_stats = calc_stats_array_Restruct(output_glac_acc_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_acc_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_refreeze_monthly_stats = calc_stats_array_Restruct(output_glac_refreeze_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_refreeze_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_melt_monthly_stats = calc_stats_array_Restruct(output_glac_melt_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_melt_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_frontalablation_monthly_stats = calc_stats_array_Restruct(output_glac_frontalablation_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_frontalablation_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_massbalclim_monthly_stats = calc_stats_array_Restruct(output_glac_massbalclim_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_massbalclim_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_massbaltotal_monthly_stats = calc_stats_array_Restruct(output_glac_massbaltotal_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_massbaltotal_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_snowline_monthly_stats = calc_stats_array_Restruct(output_glac_snowline_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_snowline_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_glac_mass_change_ignored_annual_stats = calc_stats_array_Restruct(output_glac_mass_change_ignored_annual,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_mass_change_ignored_annual',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_offglac_prec_monthly_stats = calc_stats_array_Restruct(output_offglac_prec_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'offglac_prec_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_offglac_melt_monthly_stats = calc_stats_array_Restruct(output_offglac_melt_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'offglac_melt_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_offglac_refreeze_monthly_stats = calc_stats_array_Restruct(output_offglac_refreeze_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'offglac_refreeze_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                output_offglac_snowpack_monthly_stats = calc_stats_array_Restruct(output_offglac_snowpack_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'offglac_snowpack_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)

                                if Dynamic_step_Monthly:
                                    output_glac_length_monthly_stats = calc_stats_array_Restruct(output_glac_length_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_length_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                                    output_glac_length_change_monthly_stats = calc_stats_array_Restruct(output_glac_length_change_monthly,rgiid_ind = rgiid,reg_id=reg_str,glacier_id=glacier_str,object_name = 'glac_length_change_monthly',traceback_fp =traceback_fp,warning_fp = warning_fp)
                            #TODO save the all statistic information of the model output, mean, std, 2.5%,25%,median,75%,97.5%,mad
                             # Compute statistics for each variable
                            stats_dict_ALL = {
                                'glac_runoff_monthly': output_glac_runoff_monthly_stats,
                                'glac_area_annual': output_glac_area_annual_stats,
                                'glac_length_annual': output_glac_length_annual_stats,
                                'glac_length_change_annual': output_glac_length_change_annual_stats,
                                'glac_frontalablation_annual': output_glac_frontalablation_annual_stats,
                                'glac_mass_annual': output_glac_mass_annual_stats,
                                'glac_mass_bsl_annual': output_glac_mass_bsl_annual_stats,
                                'glac_ELA_annual': output_glac_ELA_annual_stats,
                                'offglac_runoff_monthly': output_offglac_runoff_monthly_stats,
                                'glac_massbaltotal_annual': output_glac_massbaltotal_annual_stats,
                                'glac_massbalclim_annual': output_glac_massbalclim_annual_stats,
                            }

                            # Add extra variables if applicable
                            if pygem_prms.export_extra_vars:
                                extra_stats = {
                                    'glac_temp_monthly': output_glac_temp_monthly_stats,
                                    'glac_prec_monthly': output_glac_prec_monthly_stats,
                                    'glac_acc_monthly': output_glac_acc_monthly_stats,
                                    'glac_refreeze_monthly': output_glac_refreeze_monthly_stats,
                                    'glac_melt_monthly': output_glac_melt_monthly_stats,
                                    'glac_frontalablation_monthly': output_glac_frontalablation_monthly_stats,
                                    'glac_massbaltotal_monthly': output_glac_massbaltotal_monthly_stats,
                                    'glac_massbalclim_monthly': output_glac_massbalclim_monthly_stats,
                                    'glac_snowline_monthly': output_glac_snowline_monthly_stats,
                                    'glac_mass_change_ignored_annual': output_glac_mass_change_ignored_annual_stats,
                                    'offglac_prec_monthly': output_offglac_prec_monthly_stats,
                                    'offglac_melt_monthly': output_offglac_melt_monthly_stats,
                                    'offglac_refreeze_monthly': output_offglac_refreeze_monthly_stats,
                                    'offglac_snowpack_monthly': output_offglac_snowpack_monthly_stats,
                                }
                                stats_dict_ALL.update(extra_stats)

                                if Dynamic_step_Monthly:
                                    dynamic_stats = {
                                        'glac_length_monthly': output_glac_length_monthly_stats,
                                        'glac_length_change_monthly': output_glac_length_change_monthly_stats,
                                    }
                                    stats_dict_ALL.update(dynamic_stats)

                            # Save all statistics to dataset
                            output_ds_all_stats_ALL = save_all_statistics(output_ds_all_stats_ALL, stats_dict_ALL, pygem_prms, Dynamic_step_Monthly)

                            # Output Mean
                            output_ds_all_stats['glac_runoff_monthly'].values[0,:] = output_glac_runoff_monthly_stats[:,0]
                            output_ds_all_stats['glac_area_annual'].values[0,:] = output_glac_area_annual_stats[:,0]
                            output_ds_all_stats['glac_length_annual'].values[0,:] = output_glac_length_annual_stats[:,0]
                            output_ds_all_stats['glac_length_change_annual'].values[0,:] = output_glac_length_change_annual_stats[:,0]
                            output_ds_all_stats['glac_frontalablation_annual'].values[0,:] = output_glac_frontalablation_annual_stats[:,0]
                            output_ds_all_stats['glac_massbaltotal_annual'].values[0,:] = output_glac_massbaltotal_annual_stats[:,0]
                            output_ds_all_stats['glac_massbalclim_annual'].values[0,:] = output_glac_massbalclim_annual_stats[:,0]
                            output_ds_all_stats['glac_mass_annual'].values[0,:] = output_glac_mass_annual_stats[:,0]
                            output_ds_all_stats['glac_mass_bsl_annual'].values[0,:] = output_glac_mass_bsl_annual_stats[:,0]
                            output_ds_all_stats['glac_ELA_annual'].values[0,:] = output_glac_ELA_annual_stats[:,0]
                            output_ds_all_stats['offglac_runoff_monthly'].values[0,:] = output_offglac_runoff_monthly_stats[:,0]
                            if pygem_prms.export_extra_vars:
                                output_ds_all_stats['glac_temp_monthly'].values[0,:] = output_glac_temp_monthly_stats[:,0] + 273.15
                                output_ds_all_stats['glac_prec_monthly'].values[0,:] = output_glac_prec_monthly_stats[:,0]
                                output_ds_all_stats['glac_acc_monthly'].values[0,:] = output_glac_acc_monthly_stats[:,0]
                                output_ds_all_stats['glac_refreeze_monthly'].values[0,:] = output_glac_refreeze_monthly_stats[:,0]
                                output_ds_all_stats['glac_melt_monthly'].values[0,:] = output_glac_melt_monthly_stats[:,0]
                                output_ds_all_stats['glac_frontalablation_monthly'].values[0,:] = (
                                        output_glac_frontalablation_monthly_stats[:,0])
                                output_ds_all_stats['glac_massbalclim_monthly'].values[0,:] = (
                                        output_glac_massbalclim_monthly_stats[:,0])
                                output_ds_all_stats['glac_massbaltotal_monthly'].values[0,:] = (
                                        output_glac_massbaltotal_monthly_stats[:,0])
                                output_ds_all_stats['glac_snowline_monthly'].values[0,:] = output_glac_snowline_monthly_stats[:,0]
                                
                                output_ds_all_stats['glac_mass_change_ignored_annual'].values[0,:] = (
                                        output_glac_mass_change_ignored_annual_stats[:,0])
                                
                                output_ds_all_stats['offglac_prec_monthly'].values[0,:] = output_offglac_prec_monthly_stats[:,0]
                                output_ds_all_stats['offglac_melt_monthly'].values[0,:] = output_offglac_melt_monthly_stats[:,0]
                                output_ds_all_stats['offglac_refreeze_monthly'].values[0,:] = output_offglac_refreeze_monthly_stats[:,0]
                                output_ds_all_stats['offglac_snowpack_monthly'].values[0,:] = output_offglac_snowpack_monthly_stats[:,0]

                                if Dynamic_step_Monthly:
                                    output_ds_all_stats['glac_length_monthly'].values[0,:] = output_glac_length_monthly_stats[:,0]
                                    output_ds_all_stats['glac_length_change_monthly'].values[0,:] = (output_glac_length_change_monthly_stats[:,0])  
                            
                            # Output median absolute deviation
                            if pygem_prms.sim_iters > 1:
                                output_ds_all_stats['glac_runoff_monthly_mad'].values[0,:] = output_glac_runoff_monthly_stats[:,1]
                                output_ds_all_stats['glac_area_annual_mad'].values[0,:] = output_glac_area_annual_stats[:,1]
                                output_ds_all_stats['glac_length_annual_mad'].values[0,:] = output_glac_length_annual_stats[:,1]
                                output_ds_all_stats['glac_length_change_annual_mad'].values[0,:] = output_glac_length_change_annual_stats[:,1]
                                output_ds_all_stats['glac_frontalablation_annual_mad'].values[0,:] = output_glac_frontalablation_annual_stats[:,1]
                                output_ds_all_stats['glac_massbaltotal_annual_mad'].values[0,:] = output_glac_massbaltotal_annual_stats[:,1]
                                output_ds_all_stats['glac_massbalclim_annual_mad'].values[0,:] = output_glac_massbalclim_annual_stats[:,1]
                                output_ds_all_stats['glac_mass_annual_mad'].values[0,:] = output_glac_mass_annual_stats[:,1]
                                output_ds_all_stats['glac_mass_bsl_annual_mad'].values[0,:] = output_glac_mass_bsl_annual_stats[:,1]
                                output_ds_all_stats['glac_ELA_annual_mad'].values[0,:] = output_glac_ELA_annual_stats[:,1]
                                output_ds_all_stats['offglac_runoff_monthly_mad'].values[0,:] = output_offglac_runoff_monthly_stats[:,1]
                                if pygem_prms.export_extra_vars:
                                    output_ds_all_stats['glac_temp_monthly_mad'].values[0,:] = output_glac_temp_monthly_stats[:,1]
                                    output_ds_all_stats['glac_prec_monthly_mad'].values[0,:] = output_glac_prec_monthly_stats[:,1]
                                    output_ds_all_stats['glac_acc_monthly_mad'].values[0,:] = output_glac_acc_monthly_stats[:,1]
                                    output_ds_all_stats['glac_refreeze_monthly_mad'].values[0,:] = output_glac_refreeze_monthly_stats[:,1]
                                    output_ds_all_stats['glac_melt_monthly_mad'].values[0,:] = output_glac_melt_monthly_stats[:,1]
                                    output_ds_all_stats['glac_frontalablation_monthly_mad'].values[0,:] = (
                                            output_glac_frontalablation_monthly_stats[:,1])
                                    output_ds_all_stats['glac_massbalclim_monthly_mad'].values[0,:] = (
                                            output_glac_massbalclim_monthly_stats[:,1])
                                    output_ds_all_stats['glac_massbaltotal_monthly_mad'].values[0,:] = (
                                            output_glac_massbaltotal_monthly_stats[:,1])
                                    output_ds_all_stats['glac_snowline_monthly_mad'].values[0,:] = output_glac_snowline_monthly_stats[:,1]
                                    output_ds_all_stats['glac_mass_change_ignored_annual_mad'].values[0,:] = (
                                            output_glac_mass_change_ignored_annual_stats[:,1])
                                    output_ds_all_stats['offglac_prec_monthly_mad'].values[0,:] = output_offglac_prec_monthly_stats[:,1]
                                    output_ds_all_stats['offglac_melt_monthly_mad'].values[0,:] = output_offglac_melt_monthly_stats[:,1]
                                    output_ds_all_stats['offglac_refreeze_monthly_mad'].values[0,:] = output_offglac_refreeze_monthly_stats[:,1]
                                    output_ds_all_stats['offglac_snowpack_monthly_mad'].values[0,:] = output_offglac_snowpack_monthly_stats[:,1]
                                    if Dynamic_step_Monthly:
                                        output_ds_all_stats['glac_length_monthly_mad'].values[0,:] = output_glac_length_monthly_stats[:,1]
                                        output_ds_all_stats['glac_length_change_monthly_mad'].values[0,:] = output_glac_length_change_monthly_stats[:,1]
                            print("------------- record 3 finish -------------")
                        except Exception:
                                stats_tl.log_traceback(object_name=None, traceback_fp=traceback_fp)
                                print(traceback.format_exc())


                        #pdb.set_trace()
                        # visualize the statistics of lengthchange and frontal ablation
                        SIM_ITERATIONS = pygem_prms.sim_iters
                        save_name_lenthchange = 'lengthchange_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.png'
                        save_name_frontalablation = 'frontalablation_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.png'
                        #pdb.set_trace()
                        # plot_timeseries_stats(output_glac_length_change_annual,start_date = 2000,end_date = 2102,save_path=save_path_figure,save_name=save_name_lenthchange)
                        # plot_timeseries_stats(output_glac_frontalablation_annual,start_date = 2000,end_date = 2102,save_path=save_path_figure,save_name=save_name_frontalablation)

                        save_name_lenthchange_sub = 'lengthchange_sub_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.png'
                        save_name_frontalablation_sub = 'frontalablation_sub_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.png'
                        #pdb.set_trace()
                        # plot_timeseries_stats_sub(output_glac_length_change_annual,start_date = 2000,end_date = 2102,save_path=save_path_figure,save_name=save_name_lenthchange_sub)
                        # plot_timeseries_stats_sub(output_glac_frontalablation_annual,start_date = 2000,end_date = 2102,save_path=save_path_figure,save_name=save_name_frontalablation_sub)

                        # save the data to csv file
                        #pdb.set_trace()
                        # TODO Check if this is necessary, since we already has the nc file, if not necessary, remove this part
                        output_glac_length_change_annual_df = pd.DataFrame(output_glac_length_change_annual.T,columns = year_values)
                        output_glac_frontalablation_annual_df = pd.DataFrame(output_glac_frontalablation_annual.T,columns = year_values)
                        output_glac_massbalclim_annual_df = pd.DataFrame(output_glac_massbalclim_annual.T,columns = year_values)
                        output_glac_massbaltotal_annual_df = pd.DataFrame(output_glac_massbaltotal_annual.T,columns = year_values)
                        output_glac_length_change_annual_df.to_csv(save_path_modeloutput + 'lengthchange_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.csv')
                        output_glac_frontalablation_annual_df.to_csv(save_path_modeloutput + 'frontalablation_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.csv')
                        output_glac_massbalclim_annual_df.to_csv(save_path_modeloutput + 'massbalclim_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.csv')
                        output_glac_massbaltotal_annual_df.to_csv(save_path_modeloutput + 'massbaltotal_' + glacier_str + '_' + gcm_name + '_' + scenario + '_'  + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '.csv')
                        #pdb.set_trace()
                    
                        # Export statistics to netcdf
                        output_sim_fp = pygem_prms.output_sim_fp + reg_str + '/' + gcm_name + '/'
                        if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                            output_sim_fp += scenario + '/'
                        output_sim_fp += 'stats/'
                        # Create filepath if it does not exist
                        if os.path.exists(output_sim_fp) == False:
                            os.makedirs(output_sim_fp, exist_ok=True)
                        # Netcdf filename
                        if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
                            # Filename
                            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba' +
                                          str(pygem_prms.option_bias_adjustment) + '_' +  str(SIM_ITERATIONS) + 'sets' + '_' +
                                          str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '_all.nc')
                        elif realization is not None:
                            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' + realization + '_' +
                                          str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                          '_' + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                          str(args.gcm_endyear) + '_all.nc')
                            np.savetxt(output_sim_fp + 'tas_mon_' + glacier_str + '_' + gcm_name + '_' + scenario + '_' + realization + '_' +
                                          str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                          '_' + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                          str(args.gcm_endyear) + '.csv', gcm_temp_adj, delimiter="\n")
                            np.savetxt(output_sim_fp + 'pr_mon_' + glacier_str + '_' + gcm_name + '_' + scenario + '_' + realization + '_' +
                                          str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                          '_' + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                          str(args.gcm_endyear) + '.csv', gcm_prec_adj, delimiter="\n")
                        else:
                            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' +
                                          str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                          '_' + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                          str(args.gcm_endyear) + '_all.nc')
                        # Export netcdf
                        output_ds_all_stats.to_netcdf(output_sim_fp + netcdf_fn, encoding=encoding) 
                        output_ds_all_stats_ALL.to_netcdf(output_sim_fp + netcdf_fn.replace('_all.nc','_ALL_STATIS.nc'), encoding=encoding_ALL)
                        
                        # Close datasets
                        output_ds_all_stats.close()
                        output_ds_all_stats_ALL.close()
                    
    
                    # ----- DECADAL ICE THICKNESS STATS FOR OVERDEEPENINGS -----
                    if pygem_prms.export_binned_thickness and glacier_rgi_table.Area > pygem_prms.export_binned_area_threshold:
                        try:
                        
                            # Distance from top of glacier downglacier
                            output_glac_bin_dist = np.arange(nfls[0].nx) * nfls[0].dx_meter
                            # Create dataset to export
                            output_ds_binned_stats, encoding_binned = (
                                    create_xrdataset_binned_stats(glacier_rgi_table, dates_table, surface_h_initial,
                                                                output_glac_bin_mass_annual,
                                                                output_glac_bin_icethickness_annual, 
                                                                output_glac_bin_massbalclim_monthly,
                                                                output_glac_bin_massbalclim_annual,
                                                                output_glac_bin_dist))
                            # Export statistics to netcdf
                            output_sim_binned_fp = pygem_prms.output_sim_fp + reg_str + '/' + gcm_name + '/'
                            if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                                output_sim_binned_fp += scenario + '/'
                            output_sim_binned_fp += 'binned/'
                            # Create filepath if it does not exist
                            if os.path.exists(output_sim_binned_fp) == False:
                                os.makedirs(output_sim_binned_fp, exist_ok=True)
                            # Netcdf filename
                            if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
                                # Filename
                                netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba' +
                                            str(pygem_prms.option_bias_adjustment) + '_' +  str(SIM_ITERATIONS) + 'sets' + '_' +
                                            str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '_binned.nc')
                            elif realization is not None:
                                netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' + realization + '_' +
                                            str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                            '_' + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                            str(args.gcm_endyear) + '_binned.nc')
                            else:
                                netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' +
                                            str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                            '_' + str(SIM_ITERATIONS) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
                                            str(args.gcm_endyear) + '_binned.nc')
                            # Export netcdf
                            output_ds_binned_stats.to_netcdf(output_sim_binned_fp + netcdf_fn, encoding=encoding_binned)
                
                            # Close datasets
                            output_ds_binned_stats.close()
                            print("------------- record 4 finish -------------")
                        except:
                            with open(traceback_fp, "a") as traceback_file:  # Change "w" to "a" for appending
                                traceback_file.write(traceback.format_exc() + "\n")  # Add a newline for clarity
                            print(traceback.format_exc())

                        
    #                    # ----- INDIVIDUAL RUNS (area, volume, fixed-gauge runoff) -----
    #                    # Create empty annual dataset
    #                    output_ds_essential_sims, encoding_essential_sims = (
    #                            create_xrdataset_essential_sims(glacier_rgi_table, dates_table))
    #                    output_ds_essential_sims['glac_area_annual'].values[0,:,:] = output_glac_area_annual
    #                    output_ds_essential_sims['glac_volume_annual'].values[0,:,:] = output_glac_volume_annual
    #                    output_ds_essential_sims['fixed_runoff_monthly'].values[0,:,:] = (
    #                            output_glac_runoff_monthly + output_offglac_runoff_monthly)
    #        
    #                    # Export to netcdf
    #                    output_sim_essential_fp = pygem_prms.output_sim_fp + reg_str + '/' + gcm_name + '/'
    #                    if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
    #                        output_sim_essential_fp += scenario + '/'
    #                    output_sim_essential_fp += 'essential/'
    #                    # Create filepath if it does not exist
    #                    if os.path.exists(output_sim_essential_fp) == False:
    #                        os.makedirs(output_sim_essential_fp, exist_ok=True)
    #                    # Netcdf filename
    #                    if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
    #                        # Filename
    #                        netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba' +
    #                                      str(pygem_prms.option_bias_adjustment) + '_' +  str(sim_iters) + 'sets' + '_' +
    #                                      str(args.gcm_bc_startyear) + '_' + str(args.gcm_endyear) + '_annual.nc')
    #                    else:
    #                        netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' +
    #                                      str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
    #                                      '_' + str(sim_iters) + 'sets' + '_' + str(args.gcm_bc_startyear) + '_' + 
    #                                      str(args.gcm_endyear) + '_annual.nc')
    #                    # Export netcdf
    #                    output_ds_essential_sims.to_netcdf(output_sim_essential_fp + netcdf_fn, encoding=encoding_essential_sims)
    #                    # Close datasets
    #                    output_ds_essential_sims.close()
                    
                    
        # print('\n\nADD BACK IN EXCEPTION\n\n')
        
        except Exception as err:
            # LOG FAILURE
            with open(txt_fp_fail, "a") as text_file:
                text_file.write(glacier_str + f' failed to complete simulation: {err}')
            # Traceback
            stats_tl.log_traceback(object_name=glacier_str, traceback_fp=traceback_fp)
            print(traceback.format_exc())

    # Global variables for Spyder development
    if not args.option_parallels:
        global main_vars
        main_vars = inspect.currentframe().f_locals


# %% GET GLACIER NUMBERS
# the function to get the glacier numbers based on the input arguments
def get_glacier_numbers(args):
    """    Get glacier numbers based on the input arguments.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    Returns:
        list: List of glacier numbers.
    """
    if args.rgi_glac_number:
        return [args.rgi_glac_number]
    elif args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'r') as f:
            return json.load(f)
    elif args.rgi_region01:
        return get_glaciers_from_calibration_region(args)
    elif pygem_prms.glac_no is not None:
        return pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=pygem_prms.rgi_regionsO1,
            rgi_regionsO2=pygem_prms.rgi_regionsO2,
            rgi_glac_number=pygem_prms.rgi_glac_number,
            glac_no=pygem_prms.glac_no,
            include_landterm=pygem_prms.include_landterm,
            include_laketerm=pygem_prms.include_laketerm,
            include_tidewater=pygem_prms.include_tidewater,
            min_glac_area_km2=pygem_prms.min_glac_area_km2
        )
        return list(main_glac_rgi_all['rgino_str'].values)

        
# the function to get the glacier numbers from the calibration results for region
def get_glaciers_from_calibration_region(args, cali_dataset_fp= None):
    """Get glacier numbers from the calibration results for a specific region.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
        cali_dataset_fp (str): File path to the calibration datasets. If None, uses default path.
    Returns:
        list: List of glacier numbers for the specified region.
    """
    # Set default calibration dataset file path if not provided
    if cali_dataset_fp is None:
        cali_dataset_fp = os.path.join(pygem_prms.hugonnet_fp, pygem_prms.hugonnet_fn)
    # Read calibration datasets, get the glacier numbers, which are splitted into several tasks, as csv file, and get the RGIId info, to get the glacier numbers
    # print('cali_dataset_fp: ', cali_dataset_fp)
    rgiid_rasks_cali = pd.read_csv(cali_dataset_fp)
    # print('rgiid_rasks_cali: ', rgiid_rasks_cali)
    rgiid_rasks_cali = rgiid_rasks_cali['RGIId'].values.flatten().tolist()
    glac_no_reg_wdata_Cali_dataset = sorted ([str(int(rgiid.split('-')[1].split('.')[0])) + '.' + rgiid.split('-')[1].split('.')[1] for rgiid in rgiid_rasks_cali])

    # Read the calibration results file, get the corresponding glacier numbers
    calibration_result_fn = output_fp_cali + 'Summary/' + str(args.rgi_region01) + '-calving_cal_ind.csv'
    rgiid_reg_wdata_Cali_all = pd.read_csv(calibration_result_fn)
    rgiid_reg_wdata_Cali = rgiid_reg_wdata_Cali_all.dropna(subset=['Neff_k'])
    rgiid_reg_wdata_Cali = rgiid_reg_wdata_Cali['RGIId'].values.flatten().tolist()
    glac_no_reg_wdata_Cali_result = sorted([str(int(rgiid.split('-')[1].split('.')[0])) + '.' + rgiid.split('-')[1].split('.')[1] for rgiid in rgiid_reg_wdata_Cali])

    # Get the intersection of the glacier numbers from the calibration dataset and the calibration results
    glac_no_reg_wdata_Cali = list(set(glac_no_reg_wdata_Cali_dataset) & set(glac_no_reg_wdata_Cali_result))

    glac_no_lsts = glac_no_reg_wdata_Cali
    #print("the glac_no_lsts is :", glac_no_lsts)
    return glac_no_lsts


# Function to set up parallel processing
def setup_parallel_processing(args):
    """Set up parallel processing based on the command line arguments.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    Returns:
        int: Number of processes to use for parallel processing.
    """
    if args.option_parallels:
        proc_count = cpu_count()
        print(f"There are {proc_count} processors available.")
        return max(1, proc_count - proc_count // 2)  # Ensure at least one process
    return 1  # Default to single processing


# Function to prepare the GCM list based on command line arguments
def prepare_gcm_list(args):
    """Prepare the list of GCMs to process based on command line arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple: A tuple containing the list of GCM names and the scenario.
    """
    # Initialize scenario
    scenario = args.scenario

    # Check if a single GCM name was provided
    if args.gcm_name:
        return [args.gcm_name], scenario

    # Check if a reference GCM name is being used
    if args.gcm_list_fn == pygem_prms.ref_gcm_name:
        return [pygem_prms.ref_gcm_name], scenario

    # Read GCMs from file
    with open(args.gcm_list_fn, 'r') as gcm_fn:
        gcm_list = gcm_fn.read().splitlines()

    # Determine the scenario if it hasn't been set
    if scenario is None:
        scenario = os.path.basename(args.gcm_list_fn).split('_')[1]

    #print(f'Found {len(gcm_list)} GCMs to process.')
    return gcm_list, scenario


# Function to prepare realizations based on command line arguments
def prepare_realizations(args):
    """Prepare the list of realizations to process based on command line arguments.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    Returns:
        list: List of realizations to process, or None if not specified.
    """
    if args.realization is not None:
        return [args.realization]
    elif args.realization_list is not None:
        with open(args.realization_list, 'r') as real_fn:
            realizations = list(real_fn.read().splitlines())
            print(f'Found {len(realizations)} realizations to process.')
            return realizations
    return None


# Function to pack variables for parallel processing
def pack_variables(glac_no, gcm_name, realizations, scenario):
    """Pack variables for parallel processing.

    Args:
        glac_no (list): List of Glacier number.
        gcm_name (str): Name of the GCM.
        realizations (list): List of realizations.
        scenario (str): Scenario associated with the processing.

    Returns:
        list: List of packed variables for parallel processing.
    """
    list_packed_vars = []
    
    if realizations is not None:
        for realization in realizations:
            for glacier in glac_no:  # Iterate over each glacier number
                list_packed_vars.append([[glacier], gcm_name, realization, scenario])
    else:
        for glacier in glac_no:
            list_packed_vars.append([[glacier], gcm_name, None, scenario])

    return list_packed_vars


# Function to process packed variables
def process_packed_variables(list_packed_vars, num_cores, fail_log):
    if num_cores > 1:
        # Parallel processing if num_cores > 1
        for n in range(len(list_packed_vars)):
            try:
                simu_MB_FA(list_packed_vars[n], num_cores=num_cores)
            except Exception as e:
                error_message = f"Error processing glacier {list_packed_vars[n]}: {e}\n"
                print(error_message)
                fail_log.write(error_message)  # Log the error
    else:
        # Sequential processing for single core
        for n in range(len(list_packed_vars)):
            try:
                simu_MB_FA(list_packed_vars[n], num_cores=1)
            except Exception as e:
                error_message = f"Error processing glacier {list_packed_vars[n]}: {e}\n"
                print(error_message)
                fail_log.write(error_message)  # Log the error


#%% main function to run the model
def main():
    # Start timing the execution
    time_start = time.time()
    
    # Set up argument parser and parse command-line arguments
    parser = getparser()
    args = parser.parse_args()

    # Set debug mode
    debug = args.debug == 1

    # Set GCM start and end years
    pygem_prms.gcm_startyear = args.gcm_startyear if args.gcm_startyear is not None else 2000
    pygem_prms.gcm_endyear = args.gcm_endyear if args.gcm_endyear is not None else 2100

    # Validate the hugonnet_fn parameter
    if args.hugonnet_fn is None:
        print("No hugonnet_fn provided. Please provide the proper hugonnet_fn.")
        exit(1)
    pygem_prms.hugonnet_fn = f"{args.hugonnet_fn}"

    # Initialize model parameters if not already set
    if 'pygem_modelprms' not in cfg.BASENAMES:
        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # Get glacier numbers based on input arguments
    glac_no = get_glacier_numbers(args)

    # Set number of cores for parallel processing
    num_cores = setup_parallel_processing(args)

    # Prepare GCMs and scenarios
    gcm_list, scenario = prepare_gcm_list(args)

    # Prepare realizations
    realizations = prepare_realizations(args)

    # Open a file for logging failed processes
    failed_txt_fp = os.path.join(pygem_prms.output_sim_fp, 'failed', str(args.rgi_region01), 'failed_info.txt')
    os.makedirs(os.path.dirname(failed_txt_fp), exist_ok=True)

    with open(failed_txt_fp, 'a') as fail_log:  # Open the file in append mode
        for gcm_name in gcm_list:
            print(f'Processing: {gcm_name} with scenario: {scenario}')
            
            # Pack variables for multiprocessing
            list_packed_vars = pack_variables(glac_no, gcm_name, realizations, scenario)

            # print('Length of packed variables:', len(list_packed_vars))
            # print('list_packed_vars:', list_packed_vars)
            # set a breakpoint for debugging, and just stop the whole process and get out
            # sys.exit()
            # Process the packed variables
            process_packed_variables(list_packed_vars, num_cores, fail_log)

    print('Total processing time:', time.time() - time_start, 's')



# #%% PARALLEL PROCESSING
# def main():
#     time_start = time.time()
#     parser = getparser()
#     args = parser.parse_args()
    
#     if args.debug == 1:
#         debug = True
#     else:
#         debug = False
#     #TODO check with input args
#     debug = True

#     # reset the gcm start and end year
#     if args.gcm_startyear is None:
#         args.gcm_startyear = 2000
#     if args.gcm_endyear is None:
#         args.gcm_endyear = 2100
#     pygem_prms.gcm_startyear = args.gcm_startyear
#     pygem_prms.gcm_endyear = args.gcm_endyear

#     # observation datasets
#     if args.hugonnet_fn is None:
#         print("No hugonnet_fn provided. Please provide the proper hugonnet_fn.")
#         exit(1)
#     else:
#         hugonnet_fn = f"{args.hugonnet_fn}"
#         pygem_prms.hugonnet_fn = hugonnet_fn


#     if not 'pygem_modelprms' in cfg.BASENAMES:
#         cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

#     # RGI glacier number
#     if args.rgi_glac_number:
#         glac_no = [args.rgi_glac_number]
#     elif args.rgi_glac_number_fn is not None:
#         with open(args.rgi_glac_number_fn, 'r') as f:
#             glac_no = json.load(f)
#     elif args.rgi_region01:
#         # read the glacier numbers from the calibration results, and remove the failed ones
#         calibration_result_fn = output_fp_cali + 'Summary/'+str(args.rgi_region01)+'-calving_cal_ind.csv'
#         rgiid_reg_wdata_Cali_all = pd.read_csv(calibration_result_fn)
#         rgiid_reg_wdata_Cali = rgiid_reg_wdata_Cali_all.dropna(subset=['Neff_k'])
#         rgiid_reg_wdata_Cali = rgiid_reg_wdata_Cali['RGIId'].values.flatten().tolist()
#         # remove the row if the value of Neff_k is nan
#         glacno_reg_wdata_Cali = sorted([(str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
#                                                  rgiid.split('-')[1].split('.')[1]) for rgiid in rgiid_reg_wdata_Cali])
#         #print("glacno_reg_wdata_Cali:", glacno_reg_wdata_Cali)
#         #sys.exit()
#         main_glac_rgi_all = modelsetup.selectglaciersrgitable(
#                 rgi_regionsO1=[args.rgi_region01], rgi_regionsO2=pygem_prms.rgi_regionsO2,
#                 rgi_glac_number=pygem_prms.rgi_glac_number, glac_no= glacno_reg_wdata_Cali,
#                 include_landterm=pygem_prms.include_landterm, include_laketerm=pygem_prms.include_laketerm, 
#                 include_tidewater=pygem_prms.include_tidewater, 
#                 min_glac_area_km2=pygem_prms.min_glac_area_km2)        
#         glac_no = list(main_glac_rgi_all['rgino_str'].values)
#         print("glac_no:", glac_no)
#         #sys.exit() 
#     elif pygem_prms.glac_no is not None:
#         glac_no = pygem_prms.glac_no
#     else:
#         main_glac_rgi_all = modelsetup.selectglaciersrgitable(
#                 rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
#                 rgi_glac_number=pygem_prms.rgi_glac_number, glac_no=pygem_prms.glac_no,
#                 include_landterm=pygem_prms.include_landterm, include_laketerm=pygem_prms.include_laketerm, 
#                 include_tidewater=pygem_prms.include_tidewater, 
#                 min_glac_area_km2=pygem_prms.min_glac_area_km2)
#         glac_no = list(main_glac_rgi_all['rgino_str'].values)

#     # Number of cores for parallel processing
#     if args.option_parallels:
#         #num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
#         proc_count = cpu_count()
#         print(f"There are {proc_count} processors are available")
#         proc_count_RT = max(1, proc_count - proc_count//2)  # Ensure at least one process
#         num_cores = proc_count_RT
#     else:
#         num_cores = 1

#     # Glacier number lists to pass for parallel processing
#     glac_no_lsts = modelsetup.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

#     # Read GCM names from argument parser
#     #gcm_name = args.gcm_list_fn
#     if args.gcm_name is not None:
#         gcm_list = [args.gcm_name]
#         scenario = args.scenario
#     elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
#         gcm_list = [pygem_prms.ref_gcm_name]
#         scenario = args.scenario
#     else:
#         with open(args.gcm_list_fn, 'r') as gcm_fn:
#             gcm_list = gcm_fn.read().splitlines()
#             scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
#             print('Found %d gcms to process'%(len(gcm_list)))
  
#     # Read realizations from argument parser
#     if args.realization is not None:
#         realizations = [args.realization]
#     elif args.realization_list is not None:
#         with open(args.realization_list, 'r') as real_fn:
#             realizations = list(real_fn.read().splitlines())
#             print('Found %d realizations to process'%(len(realizations)))
#     else:
#         realizations = None
    
#     # Producing realization or realization list. Best to convert them into the same format!
#     # Then pass this as a list or None.
#     # If passing this through the list_packed_vars, then don't go back and get from arg parser again!
#     # Open a file for logging failures, to save the failed information
#     # Construct the file path
#     failed_txt_fp = os.path.join(pygem_prms.output_sim_fp, 'failed', str(args.rgi_region01), 'failed_info.txt')
#     # Create directories if they do not exist
#     os.makedirs(os.path.dirname(failed_txt_fp), exist_ok=True) 

#     with open(failed_txt_fp, 'a') as fail_log: #Open the file in append mode
#         # Loop through all GCMs
#         for gcm_name in gcm_list:
#             if args.scenario is None:
#                 print('Processing:', gcm_name)
#             elif not args.scenario is None:
#                 print('Processing:', gcm_name, scenario)
#             # Pack variables for multiprocessing
#             list_packed_vars = []          
#             if realizations is not None:
#                 for realization in realizations:
#                     for count, glac_no_lst in enumerate(glac_no_lsts):
#                         list_packed_vars.append([count, glac_no_lst, gcm_name, realization])
#             else:
#                 for count, glac_no_lst in enumerate(glac_no_lsts):
#                     list_packed_vars.append([count, glac_no_lst, gcm_name, realizations])
                    
#             print('len list packed vars:', len(list_packed_vars))
            
#             # Parallel processing
#             if args.option_parallels:
#                 # If there's only one item in list_packed_vars, parallelize inside the item
#                 if len(list_packed_vars) == 1:
#                     data = list_packed_vars[0]
#                     try:
#                         # Parallelize across `num_cores` inside the single glacier's iterations
#                         simu_MB_FA(data, num_cores)  # This will handle parallelism inside the single glacier
#                     except Exception as e:
#                         error_message = f"Error processing {data}: {e}\n"
#                         print(error_message)
#                         fail_log.write(error_message)  # Log the error to the file
#                 else:
#                     # Parallelize across glaciers (list_packed_vars) using multiprocessing
#                     #with multiprocessing.Pool(num_cores) as p:
#                     #    p.starmap(simu_MB_FA, [(data, num_cores) for data in list_packed_vars])
#                     for n in range(len(list_packed_vars)):
#                         try:
#                             simu_MB_FA(list_packed_vars[n],num_cores = num_cores)
#                         except Exception as e:
#                             error_message = f"Error processing glacier {list_packed_vars[n]}: {e}\n"
#                             print(error_message)
#                             fail_log.write(error_message)  # Log the error to the file
#             # If not in parallel, then only should be one loop
#             else:
#                 # Loop through the chunks and export bias adjustments
#                 for n in range(len(list_packed_vars)):
#                     try:
#                         simu_MB_FA(list_packed_vars[n],num_cores = 1)
#                     except Exception as e:
#                         error_message = f"Error processing glacier {list_packed_vars[n]}: {e}\n"
#                         print(error_message)
#                         fail_log.write(error_message)  # Log the error to the file
                   



#     print('Total processing time:', time.time()-time_start, 's')



if __name__ == "__main__":
    main()
# ##%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
# #    # Place local variables in variable explorer
# #    if args.option_parallels == 0:
# #        main_vars_list = list(main_vars.keys())
# #        gcm_name = main_vars['gcm_name']
# #        main_glac_rgi = main_vars['main_glac_rgi']
# #        if pygem_prms.hyps_data in ['Huss', 'Farinotti']:
# #            main_glac_hyps = main_vars['main_glac_hyps']
# #            main_glac_icethickness = main_vars['main_glac_icethickness']
# #            main_glac_width = main_vars['main_glac_width']
# #        dates_table = main_vars['dates_table']
# #        gcm_temp = main_vars['gcm_temp']
# #        gcm_tempstd = main_vars['gcm_tempstd']
# #        gcm_prec = main_vars['gcm_prec']
# #        gcm_elev = main_vars['gcm_elev']
# #        gcm_lr = main_vars['gcm_lr']
# #        gcm_temp_adj = main_vars['gcm_temp_adj']
# #        gcm_prec_adj = main_vars['gcm_prec_adj']
# #        gcm_elev_adj = main_vars['gcm_elev_adj']
# #        gcm_temp_lrglac = main_vars['gcm_lr']
# #        ds_stats = main_vars['output_ds_all_stats']
# ##        output_ds_essential_sims = main_vars['output_ds_essential_sims']
# #        ds_binned = main_vars['output_ds_binned_stats']
# ##        modelprms = main_vars['modelprms']
# #        glacier_rgi_table = main_vars['glacier_rgi_table']
# #        glacier_str = main_vars['glacier_str']
# #        if pygem_prms.hyps_data in ['OGGM']:
# #            gdir = main_vars['gdir']
# #            fls = main_vars['fls']
# #            width_initial = fls[0].widths_m
# #            glacier_area_initial = width_initial * fls[0].dx
# #            mbmod = main_vars['mbmod']
# #            ev_model = main_vars['ev_model']
# #            diag = main_vars['diag']
# #            if pygem_prms.use_calibrated_modelparams:
# #                modelprms_dict = main_vars['modelprms_dict']
