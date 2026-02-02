"""Calibrate the four parameters (kp,tbias, ddfsnow, tau) by the AMIS methods """
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Revised by Ruitang Yang, supported by Kristoffer on 30 Dec 2024
# It's a copy of the run_calibration_AMIS_MB_FA_20002010_Parallel_Log.py, but revising the create/save way of the output csv file so that it can be extended.
#  And also debug it for the regional running
import sys
sys.path.insert(0, '/home/ruitang/PyGEM_2023')
sys.path.insert(0, '/home/ruitang/PyGEM_2023/PyGEM-scripts')
# Built-in libraries
import argparse
import collections
import copy
import inspect
import multiprocessing
import os
import sys
import time
import glob
import cftime
import traceback
import pdb
import shutil
import h5py
import json
import logging
from multiprocessing import Pool, cpu_count, current_process
from pathlib import Path # Better path handling
from scipy.special import expit, logit  # Important for transforms!
# External libraries
import pandas as pd
import pickle
import xarray as xr
import matplotlib
#matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from scipy.stats import median_abs_deviation, truncnorm, gamma, uniform, norm,lognorm
import xarray as xr
from functools import partial
from scipy import special
import ast  # To safely evaluate string representations of lists and dictionaries
from datetime import date, datetime
from scipy.stats import median_abs_deviation, skew, kurtosis
import statsmodels.stats.correlation_tools as ct
import statistic_tool as statis_tool
from statistic_tool import GlacierLogger


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
from pygem.shop import debris, mbdata, icethickness 
from pygem import class_climate
import Visualization_timeseries as Visualization_timeseries


import oggm
oggm_version = float(oggm.__version__[0:3])
from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm import workflow
if oggm_version > 1.301:
    from oggm.core.massbalance import apparent_mb_from_any_mb # Newer Version of OGGM
else:
    from oggm.core.climate import apparent_mb_from_any_mb # Older Version of OGGM
from oggm.core.flowline import FluxBasedModel, SemiImplicitModel
from oggm.core.calving_Jan_Ruitang import CalvingFluxBasedModelJanRt
from oggm.core.inversion_RT_New import find_inversion_calving_from_any_mb
#from oggm.core.inversion import find_inversion_calving_from_any_mb

cfg.PARAMS['hydro_month_nh']=1
cfg.PARAMS['hydro_month_sh']=1
#cfg.PARAMS['trapezoid_lambdas'] = 1


#%% ----- MANUAL INPUT DATA -----
#regions = [1,3,4,5,7,9,17,19]
regions = [17]
overwrite = True

# ---- Store_monthly_step ----
#store_monthly_step = True
mb_elev_feedback = 'Monthly'  # 'annual' or 'monthly'
# TODO : Add the option to store monthly step results for mass balance and glacier dynamics
Dynamic_step_Monthly = True

#%% ----- the path of output -----
output_fp = pygem_prms.main_directory + '/Calibration/'
if not os.path.exists(output_fp):
    os.makedirs(output_fp)

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

# the path to save the AMIS INFO
save_path_AMISINFO = output_fp + '/AMIS_info/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_AMISINFO):
    os.makedirs(save_path_AMISINFO)

# the path to save the log file
save_path_log = output_fp + '/log/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)
# initialize the main logger
gl_logger = GlacierLogger(save_path_log)

# the path to save the statistic info about the model run, e.g the number of successful runs, as well as the number of failures
save_path_statistics = output_fp + '/Statistics_model_run/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_statistics):
    os.makedirs(save_path_statistics)

# the path to save the summary of the model output
save_path_summary = output_fp + '/Summary/'
# Check if the directory exists, and if not, create it
if not os.path.exists(save_path_summary):
    os.makedirs(save_path_summary)

# the path to save the floating warning info
floating_info_fp = output_fp + '/Floating_Warning_Info/'
# Check if the directory exists, and if not, create it
if not os.path.exists(floating_info_fp):
    os.makedirs(floating_info_fp)

#set the log file
def setup_worker_logger(save_path_log=None,log_level=None):
    """Set up a separate logger for each worker process with a unique log file.
    
    Args:
        save_path_log (str or Path): Directory where worker logs should be stored
        log_level (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        
    Returns:
        logging.Logger: Configured logger instance for the current worker
    """
    worker_id = current_process().name  # e.g., 'ForkPoolWorker-1'
    logger = logging.getLogger(worker_id)
    
    # Return existing logger if already configured
    if logger.handlers:
        return logger
    
    # Convert to Path and ensure directory exists
    log_dir = Path(save_path_log)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create worker-specific log file
    worker_log_path = log_dir / f"worker_{worker_id}.log"
    
    # Configure handler
    handler = logging.FileHandler(
        filename=worker_log_path,
        mode='a'  # Append mode to preserve logs between runs
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Configure logger
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level))
    logger.propagate = False  # Prevent duplicate logging from root logger
    
    return logger

# Function to log floating terminus warnings
def log_floating_warning(glacier_str, log_dir= None):
    """
    Log the floating terminus warning to a separate file.
    Only this warning is logged; other prints are untouched.
    """

    # Ensure the log directory exists
    if log_dir is None:
        log_dir = save_path_log + '/Floating_Warnings/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "floating_warnings.log")

    # Create a dedicated logger for floating warnings
    logger = logging.getLogger(f"floating_logger_{glacier_str}")
    logger.setLevel(logging.WARNING)

    # Avoid adding multiple handlers if logger is reused
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Log the warning
    logger.warning(f"Glacier {glacier_str} terminus is floating")


# Function to record floating terminus information
def record_floating_terminus(glacier_str, th, thick0, water_level, rho, rho_o,index_particles,
                             floating_info_fp_glac = None):
    """
    Record floating terminus information (backward compatible version).
    Performs exactly the same operations as original code, just organized better.
    --- Parameters ---
    glacier_str (str): Glacier identifier, e.g., '17.01156'
    th (float): Current ice thickness at terminus
    thick0 (float): Initial ice thickness at terminus
    water_level (float): Water level at terminus
    rho (float): Ice density
    rho_o (float): Ocean water density
    floating_info_fp_glac (str): File path to save floating info
    index_particles (int): Index of the particle

    """
    # Early exit if no floating condition
    if glacier_str is None or th is None or thick0 is None or water_level is None or rho is None or rho_o is None or index_particles is None:
        print("Error: Missing required parameters for recording floating terminus information.")
        return
  
    buoyancy_threshold = (1 - rho / rho_o) * thick0

    if th >= buoyancy_threshold:
            return
    print(f"Warning: Terminus of glacier {glacier_str} is floating "
        f"(th={th:.2f}m < {buoyancy_threshold:.2f}m)")
    
    # Early exit if no output directory specified
    if floating_info_fp_glac is None:
        return
    else:
        # Ensure output directory exists
        os.makedirs(floating_info_fp_glac, exist_ok=True)
        iteration = 'Poster'  # Just recording posterior samples
        # Construct file path
        filepath = os.path.join(floating_info_fp_glac, f"floating_info_glacier_{glacier_str}.csv")
        # Prepare data for writing
        header = "glacier_id,iteration,particle_index,th,thick0,water_level\n"
        data_row = (f"{glacier_str},{iteration},{index_particles},"
                    f"{th:.2f},{thick0:.2f},{water_level:.2f}\n")
        
        # Write to file
        try:
            # Check if file exists to determine if header is needed
            write_header = not os.path.exists(filepath)
            
            with open(filepath, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header)
                f.write(data_row)
                
        except (IOError, OSError) as e:
            print(f"Error writing floating terminus data for {glacier_str}: {e}")



#%% ----- The boundary condition for length change myr -----
max_length_change_myr = 5000
min_length_change_myr = -5000
#%% ----- CONVERSION FUNCTIONS -----
def mwea_to_gta(mwea, area_m2):
    return mwea * pygem_prms.density_water * area_m2 / 1e12
def gta_to_mwea(gta, area_m2):
    return gta * 1e12 / pygem_prms.density_water / area_m2

# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    log_level : str
        Set the logging level (default: INFO)
    rgi_region01 (optional) : int
        Randoph Glacier Inventory region
    rgi_glac_number (optional) : str
        Randoph Glacier Inventory glacier number(ex, '11.00897')
    gcm_bc_startyear (optional) : int
        start year for bias correction
    gcm_startyear (optional) : int
        start year for the model run
    gcm_endyear (optional) : int
        end year for the model run
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
        frontalablation_fn (optional) : str
        filename of .pkl file containing period averaged frontal ablation data (observations)
    frontalablation_annual_fn (optional) : str
        filename of .pkl file containing annual timeseries frontal ablation data (observations)
    hugonnet_fn (optional) : str
        filename of .pkl file containing period averaged mass balance (climatic) data (observations)
    lengthchange_annual_fn (optional) : str
        filename of .pkl file containing annual timeseries length change data (observations)
    rgi_glac_number_fn (optional) : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
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

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
        # Logging group
    logging_group = parser.add_argument_group('Logging options')
    logging_group.add_argument('--log_level', 
                            default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Set the logging level (default: INFO)')
    logging_group.add_argument('--log-file', 
                            default=None,
                            help='Path to the log file (default: None)')
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
    parser.add_argument('-frontalablation_fn', action='store', type=str, default=None,
                        help='Filename containing period averaged frontal ablation data (observations)')
    parser.add_argument('-frontalablation_annual_fn', action='store', type=str, default=None,
                        help='Filename containing annual timeseries frontal ablation data (observations)')
    parser.add_argument('-hugonnet_fn', action='store', type=str, default=None,
                        help='Filename containing period averaged mass balance (climatic) data (observations)')
    parser.add_argument('-lengthchange_annual_fn', action='store', type=str, default=None,
                        help='Filename containing annual timeseries length change data (observations)')
    # flags
    parser.add_argument('-option_ordered', action='store_true',
                        help='Flag to keep glacier lists ordered (default is off)')
    parser.add_argument('-option_parallels', action='store_true',
                        help='Flag to use or not use parallels (default is off)')
    parser.add_argument('-debug', action='store_true',
                        help='Flag for debugging (default is off')
    parser.add_argument('-debug_spc', action='store_true',
                        help='Flag for debugging (default is off')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Flag for verbose')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Flag to overwrite existing calibrated frontal ablation datasets')  
    parser.add_argument('-store_monthly_step', action='store_true',
                        help='Flag to store the monthly step results')
    parser.add_argument('-Visualize_Index', action='store_true',
                        help='Flag to visualize the index number of the samples')
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=pygem_prms.ref_gcm_name,
                    help='reference gcm name')
    parser.add_argument('-ref_startyear', action='store', type=int, default=pygem_prms.ref_startyear,
                        help='reference period starting year for calibration (typically 2000)')
    parser.add_argument('-ref_endyear', action='store', type=int, default=pygem_prms.ref_endyear,
                        help='reference period ending year for calibration (typically 2019)')  
    
    return parser



# def convert_to_serializable(obj):
#     """
#     Recursively convert non-serializable objects to JSON-compatible formats.
#     """
#     if isinstance(obj, np.ndarray):  
#         return obj.tolist()  # Convert NumPy array to list
#     elif isinstance(obj, np.generic):  
#         return obj.item()  # Convert NumPy scalar to int/float
#     elif isinstance(obj, (date, datetime, np.datetime64)):  
#         return str(obj)  # Convert date/datetime64 to string
#     elif isinstance(obj, dict):  
#         return {k: convert_to_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, list):  
#         return [convert_to_serializable(i) for i in obj]
#     elif isinstance(obj, set):  
#         return list(obj)
#     elif isinstance(obj, tuple):  
#         return list(obj)
#     elif isinstance(obj, (int, float, str)):  
#         return obj
#     else:
#         return str(obj)  # Convert unknown objects to strings

def convert_to_serializable(obj):
    """
    Recursively convert non-serializable objects to JSON-compatible formats.
    """
    if isinstance(obj, np.ndarray):  
        return obj.tolist()  # Convert NumPy array to list
    if isinstance(obj, np.generic):  
        return obj.item()  # Convert NumPy scalar to int/float
    if isinstance(obj, (date, datetime, np.datetime64)):  
        return str(obj)  # Convert date/datetime64 to string
    if isinstance(obj, dict):  
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively convert dict
    if isinstance(obj, (list, set, tuple)):  
        return [convert_to_serializable(i) for i in obj]  # Convert iterable to list
    if isinstance(obj, (int, float, str, bool, type(None))):  
        return obj  # Keep basic types unchanged

    return str(obj)  # Fallback conversion to string

def glogit(x, xmin, xmax):
    """ Generalized logit that transforms a logit-normal to a normal"""
    xs = (x - xmin) / (xmax - xmin)  # Scale x\in(a,b) to be between (0,1) instead
    xt = logit(xs)
    return xt


def gexpit(xt, xmin, xmax):
    """ Generalized expit that transforms a normal to a logit-normal"""

    xs = expit(xt)
    x = (xmax - xmin) * xs + xmin
    return x


# The function to transform between logitnormal to normal distribution
def Transform_norm (Median, Std,Min,Max,N,name_para, Visual_Index = False):
    """
    # Example how to generate a ensemble paritcles of logit-normal prior
    Median=Median # prior median 
    Std=Std # Spread (defined in transformed space, tune this to what looks nice)
    Min=Min # Minimum value 
    Max=Max  # Max value (don't make this too small!)
    """
    Ne=N # number of particles
    ddfprit=glogit(Median,Min,Max)+Std*np.random.randn(Ne) # Transformed (normal) prior
    ddfpri=gexpit(ddfprit,Min,Max) # Actual logit-normal prior in model (PyGEM) space
    

    
    if Visual_Index:
        ddfprit_mean = glogit(Median,Min,Max)
        ddfprit_std = Std
        min_x = np.min(ddfpri)
        max_x = np.max(ddfpri)
        print(f"Type of ddfprit_mean: {type(ddfprit_mean)}")
        print(f"Shape of ddfprit_mean: {np.shape(ddfprit_mean)}")
        print("ddfprit is :",ddfprit)
        print("The mean of the normal distribution is :",ddfprit_mean)

        plt.subplot(1,2,2)
        plt.hist(ddfpri,bins=30, density=True, alpha=0.6)
        plt.xlim(min_x,max_x)
        plt.title(f"{name_para} (logit-normal)")


        plt.subplot(1,2,1)
        plt.hist(ddfprit,bins=30, density=True, alpha=0.6)
        plt.title(f" transformed {name_para} (normal)")
        plt.show()
    
    return ddfpri


# Function to sample each parameter based on its prior distribution
def sample_prior(N = 1000, Visualize = False):
    """
    Samples from the prior distributions of tbias, kp, ddfsnow, and tau, and the index number.
    Ensures the size of samples for each parameter is strictly equal to N.
    Args:
        N (int): Number of samples to generate for each parameter.
        Visualize (bool): Option to visualize the prior distributions,default is False.
    Returns:
        dict: A dictionary containing exactly N sampled values for each parameter.
    """
    # Initialize a dictionary to store samples
    samples = {}

    # --- tbias: Truncated Normal Distribution ---
    tbias_med = 0.021         # Median
    tbias_sigma = 0.25      # Standard deviation
    tbias_bndlow = -10   # Lower bound
    tbias_bndhigh = 10   # Upper bound
    samples["tbias"] = Transform_norm (tbias_med, tbias_sigma,tbias_bndlow,tbias_bndhigh,N,'Tbias',Visual_Index = Visualize)
        
    # kp
    kp_med = 2.17         # Median
    kp_sigma = 0.6      # Standard deviation
    kp_bndlow = 0   # Lower bound
    kp_bndhigh = 6   # Upper bound
    samples["kp"] = Transform_norm (kp_med, kp_sigma,kp_bndlow,kp_bndhigh,N,'kp',Visual_Index = Visualize)
    
    # ddfsnow
    ddfsnow_med = 0.0041         # Median
    ddfsnow_sigma = 0.4      # Standard deviation
    ddfsnow_bndlow = 0   # Lower bound
    ddfsnow_bndhigh = 0.02   # Upper bound
    samples["ddfsnow"] = Transform_norm (ddfsnow_med, ddfsnow_sigma,ddfsnow_bndlow,ddfsnow_bndhigh,N,'ddfsnow',Visual_Index = Visualize)
     
    # tau
    tau_med = 1.453         # Median
    tau_sigma = 0.6      # Standard deviation
    tau_bndlow = 0   # Lower bound
    tau_bndhigh = 4   # Upper bound
    samples["tau"] = Transform_norm (tau_med, tau_sigma,tau_bndlow,tau_bndhigh,N,'tau',Visual_Index = Visualize)
    
    # Index
    samples["index"]=np.arange(N)   
    
    return samples
    

# The function to convert a dictionary to a Xarray
def samples2xrDataset(samples):
    """
    Convert sample dictionary to Xarray Dataset
    Args:
        samples (dict): Dictionary containing samples.
    Returns:
        ds: Xarray Dataset.
    """
    vars=dict()
    dims=['sample_index']
    smpl=np.arange(samples['index'].size)
    coords={'sample_index':smpl}
    for key in samples.keys():
        if (key != 'index'):
            vars[key]=(['sample_index'],samples[key])
            var=xr.DataArray(data=samples[key],dims=dims,coords=coords,name=key,attrs={'units':'-', 'long_name':key,})
            vars[key]=var
    ds = xr.Dataset(data_vars=vars,coords=coords,attrs={'creation_date':str(datetime.datetime.now()),'author':'Ruitang Yang'})
    ##pdb.set_trace()
    return ds


def netcdf2samples(fname):
    """
    Read netcdf file and get sample data
    Args:
        path (str): full path to netcdf file
    Returns:
        samples (dict): Dictionary containing samples.
    """
    ds = xr.open_dataset(fname)
    samples = {}
    for key in ds.data_vars.keys():
        if key != 'sample_index':
            samples[key] = ds[key].values
    samples['index'] = ds['sample_index'].values
    return samples



# Function to transform the parameter space
def transform_space(parameters, trans_direction, pert_stra=False, vari=False):

    safe_pars = parameters.copy()
    perturbation_strategy = pygem_prms.perturbation_strategy
    upper_bounds = pygem_prms.pygem_upper_bounds
    lower_bounds = pygem_prms.pygem_lower_bounds
    vars_to_perturbate = pygem_prms.vars_to_calibrate

    # HACK: This if is to correct a single vector
    if isinstance(pert_stra, str) and isinstance(vari, str):
        perturbation_strategy = [pert_stra]
        vars_to_perturbate = [vari]

    if trans_direction == 'to_normal':
        # translate lognormal variables to normal distribution
        for cont, var in enumerate(perturbation_strategy):

            var_tmp = vars_to_perturbate[cont]

            if var == "lognormal":
                safe_pars[cont, :] = np.log(safe_pars[cont, :])

            elif var in ["logitnormal_mult",
                         "logitnormal_adi"]:

                safe_pars[cont, :] = glogit(safe_pars[cont, :],
                                                lower_bounds[var_tmp],
                                                upper_bounds[var_tmp])
            else:
                pass

    elif trans_direction == 'from_normal':

        for cont, var in enumerate(perturbation_strategy):
            var_tmp = vars_to_perturbate[cont]

            if var == "lognormal":
                safe_pars[cont, :] = np.exp(safe_pars[cont, :])

            elif var in ["logitnormal_mult",
                         "logitnormal_adi"]:

                safe_pars[cont, :] = gexpit(safe_pars[cont, :],
                                                lower_bounds[var_tmp],
                                                upper_bounds[var_tmp])
            else:
                pass

    else:
        raise Exception("transformation not found")

    return safe_pars



def pbs(obs, pred, R):
    """
    PBS: Implmentation of the Particle Batch Smoother
    Inputs:
        obs: Observation vector (m x 1 array)
        pred: Predicted observation ensemble matrix (m x N array)
        r_cov: Observation error covariance 'matrix' (m x 1 array, or scalar)
    Outputs:
        w: Posterior weights (N x 1 array)
    Dimensions:
        ens_mem is the number of ensemble members and m is the number
        of observations.

    Here we have implemented the particle batch smoother, which is
    a batch-smoother version of the particle filter (i.e. a particle filter
    without resampling), described in Margulis et al.
    (2015, doi: 10.1175/JHM-D-14-0177.1). As such, this routine can also be
    used for particle filtering with sequential data assimilation. This scheme
    is obtained by using a particle (mixture of Dirac delta functions)
    representation of the prior and applying this directly to Bayes theorem. In
    other words, it is just an application of importance sampling with the
    prior as the proposal density (importance distribution). It is also the
    same as the Generalized Likelihood Uncertainty Estimation (GLUE) technique
    (with a formal Gaussian likelihood)which is widely used in hydrology.

    This particular scheme assumes that the observation errors are additive
    Gaussian white noise (uncorrelated in time and space). The "logsumexp"
    trick is used to deal with potential numerical issues with floating point
    operations that occur when dealing with ratios of likelihoods that vary by
    many orders of magnitude.

    Based on a previous version from  K. Aalstad (14.12.2020), revised by Ruitang
    """
    # TODO: Implement an option for correlated observation errors if R is
    #      specified as a matrix (this would slow down this function).
    # TODO: Consier other likelihoods and observation models.
    # TODO: Look into iterative versions of importance sampling.

    # Check if R is a list, convert it to a numpy array if so
    R = np.array(R) if isinstance(R, list) else R
    pred = np.array(pred) if isinstance(pred, list) else pred
    # Dimensions.
    n_obs = np.size(obs)  # Number of obs
    ens_mem = np.shape(pred)[-1]

    # Checks on the observation error covariance matrix.
    if np.size(R) == 1:
        R = R * np.ones(n_obs)
    elif np.size(R) == n_obs:
        pass
    else:
        raise Exception('r_cov must be a scalar, m x 1 vector.')

    # Residual and log-likelihood
    if n_obs == 1:
        residual = obs - pred
        llh = -0.5 * ((residual**2) * (1/R))
    else:
        #pdb.set_trace()
        residual = np.array(obs) - np.array(pred)
        llh = -0.5 * (1/R.flatten()) @ (residual**2)

    # Log of normalizing constant
    # A properly scaled version of this could be output for model comparison.
    log_z = special.logsumexp(llh)  # from scipy.special import logsumexp

    # Weights
    logw = llh - log_z  # Log of posterior weights
    weights = np.exp(logw)  # Posterior weights
    
    if np.shape(weights)[-1] == ens_mem and np.round(np.sum(weights), 10) == 1:
        pass
    else:
        raise Exception('Something wrong with the PBS')

    weights = np.squeeze(weights)  # Remove axes of length one

    Neff = 1/np.sum(weights**2)
    Neff = np.round(Neff)

    return weights, Neff


# Adapted PBS ( AMIS) , Ruitang revising from the original code by K. Aalstad (22.02.2025)
def AMIS(obs, pred, R, prim, pric, propm, propc, props):
    """
    AMIS: Adaptive Multiple Importance Sampling
    Inputs:
        obs: Observation vector (No x 1 array)
        pred: Predicted observation ensemble (No x Ne x Nl array)
        R: Observation error covariance 'matrix' (No x 1 array, or scalar)
        prim: The mean (vector) of the prior (Np x 1 array)
        pric: The covariance (matrix) of the prior (Np x Np array)
        propm: Means of the DM proposal (Np x Nl)
        propc: Covariances of the DM proposal (Np x Nl)
        proposal: Samples from the DM proposal (Np x Ne x Nl array)
    Outputs:
        w: Posterior weights (Ne x 1 array)
    Dimensions
        Ne is the number of ensemble members, Np is the number of state
        variables and/or parameters, No is the number of observations, Nl
        is the number of iterations thus far.

    This AMIS scheme is based on the work of Cornuet et al. (2012) which is
    based on combining the Population Monte Carlo approach with the
    so-called deterministic mixture (DM) approach of Owen and Zhou (2000).
    It is less wasteful, more stable, and often faster to converge than
    simpler AIS algorithms.

    Code by: K. Aalstad (August 2023) based on an earlier Matlab version.
    """

    # Dimensions.
    No = np.shape(pred)[0]
    Ne = np.shape(pred)[1]
    Nl = np.shape(pred)[2]
    # Np=np.shape(propm)[0]

    # Observation covariance
    if np.size(R) == 1:
        R = R * np.ones(No)
    elif np.size(R) != No:
        raise ValueError("R must be scalar or match number of observations")

    # Initialize
    w = np.zeros(Ne)
    Neff = 0.0
    try:
        # Identify ensembles with any NaNs
        nan_mask = np.any(np.isnan(pred), axis=(0, 2))  # shape: (Ne,)
        valid_idx = ~nan_mask
        if not np.any(valid_idx):
            # All ensembles are invalid
            return w, np.nan
        # Only keep valid ensemble members for computation
        pred_valid = pred[:, valid_idx, :]
        props_valid = props[:, valid_idx, :]
        Ne_valid = pred_valid.shape[1]
        phi = np.zeros([Ne_valid, Nl])
        lsepsi = np.zeros([Ne_valid, Nl])
        # cy = np.linalg.det(2 * np.pi * np.diag(R)) ** (-0.5)
        # cy = np.linalg.det(2 * np.pi * np.diagflat(R)) ** (-0.5)
        # c0 = np.linalg.det(2 * np.pi * pric) ** (-0.5)
        # b = cy * c0

        # phi = np.zeros([Ne, Nl])  # negative log of target
        # lsepsi = np.zeros([Ne, Nl])  # logsumexp of the DM proposal
        for ell in range(Nl):
            # Terms related to the target
            propell = props_valid[:, :, ell]  # Np x Ne_valid
            A0ell = (propell.T - prim).T
            B = np.linalg.solve(pric, A0ell)
            phi0ell = 0.5 * np.sum((A0ell.T) * B.T, 1)
            predell = pred_valid[:, :, ell]  # No x Ne_valid
            residuell = (obs.flatten() - predell.T).T  # No x Ne_valid
            #residuell = (obs - predell).T  # No x Ne_valid
            #pdb.set_trace()
            phidell = 0.5 * (1 / R.flatten()) @ (residuell ** 2)  # Ne
            phi[:, ell] = phi0ell + phidell

            psij = np.zeros([Ne_valid, Nl])
            for j in range(Nl):
                mj = propm[:, j]
                Cj = propc[:, :, j]
                # --- REGULARIZATION ---
                eps = 1e-6 * np.trace(Cj) / Cj.shape[0]
                Cj = Cj + eps * np.eye(Cj.shape[0])
                sign, logdet = np.linalg.slogdet(2 * np.pi * Cj)
                if sign <= 0 or not np.isfinite(logdet):
                    raise np.linalg.LinAlgError("Invalid proposal covariance")

                lcj = -0.5 * logdet
                # cj = np.linalg.det(2 * np.pi * Cj) ** (-0.5)
                # lcj = np.log(cj)
                Aj = (propell.T - mj).T
                # B = np.linalg.solve(Cj, Aj)
                try:
                    B = np.linalg.solve(Cj, Aj)
                except np.linalg.LinAlgError:
                    B = np.linalg.lstsq(Cj, Aj, rcond=None)[0]
                psi = 0.5 * np.sum((Aj.T) * B.T, 1)
                psi = psi - lcj
                psij[:, j] = psi
            psijx = np.max(psij, axis=1)  # Ne
            psijs = (psij.T - psijx).T  # Ne x Nl
            lsepsiell = psijx + np.log(np.sum(np.exp(psijs), 1))
            lsepsi[:, ell] = lsepsiell

        # Combine terms
        logwt = -phi - lsepsi          # shape (Ne, Nl)

        # Identify valid ensemble-iteration pairs
        valid = np.isfinite(logwt)

        # Initialize weights
        w = np.zeros_like(logwt)

        if not np.any(valid):
            # All failed → return safely
            return w.flatten('F'), 0.0

        # Work only on valid entries
        logwt_valid = logwt[valid]

        # Numerically stable normalization
        lwtx = np.max(logwt_valid)
        lse = lwtx + np.log(np.sum(np.exp(logwt_valid - lwtx)))

        logNlNe = np.log(np.sum(valid))
        logZ = -logNlNe + lse

        logw_valid = logwt_valid - logNlNe - logZ

        # Assign back
        w[valid] = np.exp(logw_valid)

        # Flatten column-major (AMIS requirement)
        w = w.flatten('F')

        # Effective sample size
        Neff = 1.0 / np.sum(w**2)
        # #logwt = np.log(b) - phi - lsepsi #TODO check the location
        # #logwt = logwt.flatten('F')  # Purposely flattening column major order
        # logwt = - phi - lsepsi
        # logwt = logwt.flatten('F')  # Purposely flattening column major order
        # lwtx = np.max(logwt)
        # lselwt = lwtx + np.log(np.sum(np.exp(logwt - lwtx)))
        # logNlNe = np.log(Nl * Ne)
        # logZ = -logNlNe + lselwt  # Log model evidence
        # logw_valid = np.exp(logwt - logNlNe - logZ)
        # # Fill the weights array, zero for NaNs
        # w[valid_idx] = logw_valid
        # w[~valid_idx] = 0.0
        # # Normalize just in case
        # w_sum = np.sum(w)
        # if w_sum > 0:
        #     w /= w_sum
        # else:
        #     # All weights zero
        #     w[:] = 1.0 / Ne
        # # logw = logwt - logNlNe - logZ
        # # w = np.exp(logw)
        # #w = np.exp(logw.reshape(Ne, Nl)[:, -1])  # Now w.shape = (20,)
        # #pdb.set_trace()
        # Neff = 1 / np.sum(w ** 2)

    except Exception:
        print(traceback.format_exc())
        return np.zeros(Ne), np.nan

    return w, Neff


def reg_calving_flux(main_glac_rgi, modelprms_MB_FA, fa_glac_data_reg=None,
                     prms_from_reg_priors=False, prms_from_glac_cal=False, ignore_nan=True, debug=True,
                     invert_standard=False,
                     calc_mb_geo_correction=False, reset_gdir=True,store_monthly_step =False, Visualize_Index = False,
                     do_DA_calib_Paralle = False,save_path_figure_glac =None,log_level = 'INFO',floating_info_fp_glac=None):
    """
    Compute the calving flux for a group of glaciers # TODO Currently, it is called by single glacier
    
    Parameters
    ----------
    main_glac_rgi : pd.DataFrame
        rgi summary statistics of each glacier
    modelprms_MB_FA : dict
        model parameters for both mass balance and calving(kp, tbias, ddfsnow, ddfice,snow threshold,precgrad,tau)
    invert_standard : Boolean, default is False
    prms_from_reg_priors : Boolean
        use model parameters from regional priors
    prms_from_glac_cal : Boolean
        use model parameters from initial calibration
    store_monthly_step: Boolean
        store the monthly step of the glacier profile,
        Here is for setting whether we do monthly/annual calibration for glacier length change rate.
        The default is false, which means we do the multi-years averaged calibration for glacier front ablation.
    Visualize_Index: Boolean
        True : Visualize the timeseries of glacier profile, length change, length change rate, accumulated calving flux and calving
        False : no visualization
    do_DA_calib_Paralle: Boolean
        True: do the data assimilation calibration in a paralle computing way
        False: do the data assimilation calibration in a sequential computing way
    save_path_figure_glac: PATH
        The path to save the figure of glacier profile, length change, length change rate, accumulated calving flux and calving
        This is only used when Visualize_Index is True
    floating_info_fp_glac: PATH
        The path to the floating info file for each glacier
    log_level : str
        Set the logging level (default: INFO), can be DEBUG, INFO, WARNING, ERROR, CRITICAL
        It is used to set the log level of the logger, and control the output of the log file and the print hints
    

    Returns
    -------
    output_df : pd.DataFrame
        Dataframe containing information pertaining to each glacier's calving flux
    """    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=pygem_prms.ref_spinupyears,
            option_wateryear=pygem_prms.ref_wateryear)
    # print('dates_table is :',dates_table)
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    assert pygem_prms.ref_gcm_name in ['ERA5', 'ERA-Interim'], (
            'Error: Calibration not set up for ' + pygem_prms.ref_gcm_name)
    gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    if pygem_prms.option_ablation == 2 and pygem_prms.ref_gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                        main_glac_rgi, dates_table)
    else:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)

    # ===== CALIBRATE ALL THE GLACIERS AT ONCE =====
    #pdb.set_trace()
    # check the type of the modelprms_MB_FA
    print("Type of modelprms_MB_FA:", type(modelprms_MB_FA))
    print("Contents of modelprms_MB_FA:", repr(modelprms_MB_FA))  # Show raw format
    print("=================================================")
    #modelprms_MB_FA = ast.literal_eval(modelprms_MB_FA)  # Converts string to dictionary

    print(type(modelprms_MB_FA))  # Should be a dictionary
    print(modelprms_MB_FA)  # Print to see the content
    #pdb.set_trace()

    calving_k = modelprms_MB_FA['tau'] # calving parameter, in k_calving called calving_k, in SermQ called tau, but using the same name calving_k
    index_particles = modelprms_MB_FA['index']
    output_cns = ['RGIId', 'calving_k', 'calving_thick', 'calving_flux_Gta_inv', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics','length_change_m','length_change_rate_myr_dLdt','velocity_at_calvingfront_myr','thickness_at_calvingfront_m','width_at_calvingfront_m','volume_bsl_m3','volume_bwl_m3']
    output_df = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cns))), columns=output_cns)
    output_df['RGIId'] = main_glac_rgi.RGIId
    output_df['calving_k'] = calving_k
    output_df['calving_thick'] = np.nan
    output_df['calving_flux_Gta'] = np.nan
    output_df['oggm_dynamics'] = 0
    output_df['mb_mwea_fa_asl_lost'] = 0 
    output_df['length_change_rate_myr_dLdt'] = [None] * output_df.shape[0]  # Initialize with None
    output_df['length_change_m'] = [None] * output_df.shape[0]  # Initialize with None
    output_df['velocity_at_calvingfront_myr'] = [None] * output_df.shape[0]  # Initialize with None
    output_df['thickness_at_calvingfront_m'] = [None] * output_df.shape[0]  # Initialize with None
    output_df['width_at_calvingfront_m'] = [None] * output_df.shape[0]  # Initialize with None
    output_df['volume_bsl_m3'] = [None] * output_df.shape[0]  # Initialize with None
    output_df['volume_bwl_m3'] = [None] * output_df.shape[0]  # Initialize with None
    # ===== RUN REGRESSION CALIBRATION ===== 
    # print('============================= run reg_calving_flux =============================')
    # print('********** main glacier rgi ********** is :',main_glac_rgi)
    for nglac in np.arange(main_glac_rgi.shape[0]):
        # print("*********************** The",nglac,"glacier ***********************")  
        # print('\n',main_glac_rgi.loc[main_glac_rgi.index.values[nglac],'RGIId'])
#        if main_glac_rgi.loc[nglac,'RGIId'] in ['RGI60-09.00855']:
        
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[nglac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        if do_DA_calib_Paralle:
            k_str=glacier_str+"_"+f"{index_particles:.0f}"
            # set the suffix of each iteration
            #file_suffix = '_'+str(index_particles) #TODO we save the file_suffix in the glacier directory here, incase, we can solve the repeat download problems for each iteration and each parameters
            file_suffix = ''
        else:
            #k_str = ''
            file_suffix = ''
        try:
            gdir = single_flowline_glacier_directory_with_calving(glacier_str, 
                                                                logging_level='DEBUG',
                                                                reset=reset_gdir,k_calving_str=k_str)
        except:
            print("**********Something is wrong with single_flowline_glacier_directory_with_calving in Reg_calving_flux**********")
            print(traceback.format_exc())

        #print("contents of gdir are",os.listdir(gdir.dir))
        
        gdir.is_tidewater = True
        cfg.PARAMS['use_kcalving_for_inversion'] = True
        cfg.PARAMS['use_kcalving_for_run'] = True

        try:
            fls = gdir.read_pickle('inversion_flowlines')
            glacier_area = fls[0].widths_m * fls[0].dx_meter
            debris.debris_binned(gdir, fl_str='inversion_flowlines', ignore_debris=True)
        except:
            fls = None
            print(traceback.format_exc())
              
        # Add climate data to glacier directory
        gdir.historical_climate = {'elev': gcm_elev[nglac],
                                   'temp': gcm_temp[nglac,:],
                                   'tempstd': gcm_tempstd[nglac,:],
                                   'prec': gcm_prec[nglac,:],
                                   'lr': gcm_lr[nglac,:]}
        gdir.dates_table = dates_table
        #TODO, the calibration data could be loaded just once, and then used for different parameter sets
        # ----- load the calibration data (climatic mass balance)
        mbdata_fn = gdir.get_filepath('mb_obs')   
        with open(mbdata_fn, 'rb') as f:
            gdir.mbdata = pickle.load(f)      
        # Non-tidewater glaciers
        if not gdir.is_tidewater:
            # Load data
            mb_obs_mwea = gdir.mbdata['mb_mwea']
            mb_obs_mwea_err = gdir.mbdata['mb_mwea_err']
        # Tidewater glaciers
        #  use climatic mass balance since calving_k already calibrated separately
        else:
            assert 'mb_clim_mwea' in gdir.mbdata.keys(), 'include_frontalablation is set as true, but fontal ablation has yet to be calibrated.'
            mb_obs_mwea = gdir.mbdata['mb_clim_mwea']
            mb_obs_mwea_err = gdir.mbdata['mb_clim_mwea_err']
        
        # ----- Invert ice thickness and run simulation ------
        if (fls is not None) and (glacier_area.sum() > 0):
            
            # ----- Model parameters -----
            # Use most likely parameters from initial calibration to force the mass balance gradient for the inversion (the initial value?）
            kp_value = modelprms_MB_FA['kp']
            #print("the initial kp_value from the emulator is:",kp_value)
            tbias_value = modelprms_MB_FA['tbias']
            #print("the initial tbias value from the emulator is:",tbias_value)
            ddfsnow_value = modelprms_MB_FA['ddfsnow']
            ddfice_value = ddfsnow_value/pygem_prms.ddfsnow_iceratio
                
            # Otherwise use input parameters
            if kp_value is None:
                kp_value = pygem_prms.kp
            if tbias_value is None:
                tbias_value = pygem_prms.tbias
            
            # Set model parameters
            modelprms = {'kp': kp_value,
                         'tbias': tbias_value,
                         'ddfsnow': ddfsnow_value,
                         'ddfice': ddfice_value,
                         'tsnow_threshold':  pygem_prms.tsnow_threshold ,
                         'precgrad': pygem_prms.precgrad}              
                
            # Calving and dynamic parameters
            cfg.PARAMS['calving_k'] = calving_k
            cfg.PARAMS['inversion_calving_k'] = cfg.PARAMS['calving_k']
            
            if pygem_prms.use_reg_glena:
                glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)
                glena_idx = np.where(glena_df.O1Region == glacier_rgi_table.O1Region)[0][0]
                glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                fs = glena_df.loc[glena_idx,'fs']
            else:
                fs = pygem_prms.fs
                glen_a_multiplier = pygem_prms.glen_a_multiplier

            # set the fs = None here and read it in the inversion_RT_New.py
            fs = None    
            
            # CFL number (may use different values for calving to prevent errors)
            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number
            else:
                cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number_calving
            # ----- Mass balance model for ice thickness inversion using OGGM -----
            mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                         hindcast=pygem_prms.hindcast,
                                         debug=pygem_prms.debug_mb,
                                         debug_refreeze=pygem_prms.debug_refreeze,
                                         fls=fls, option_areaconstant=True,
                                         inversion_filter=False)
            #print("mbmod_inv is:",mbmod_inv)
            h, w = gdir.get_inversion_flowline_hw()
            #print("the surface elevation (m a.b.s.l) is:",h)
            #print("the width (m) is:",w)
            #            if debug:
            #                mb_t0 = (mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls) * cfg.SEC_IN_YEAR * 
            #                         pygem_prms.density_ice / pygem_prms.density_water) 
            #                plt.plot(mb_t0, h, '.')
            #                plt.ylabel('Elevation')
            #                plt.xlabel('Mass balance (mwea)')
            #                plt.show()
                                
            # ----- CALVING -----
            # Number of years (for OGGM's run_until_and_store)
            if pygem_prms.timestep == 'monthly':
                nyears = int(dates_table.shape[0]/12)
            else:
                assert True==False, 'Adjust nyears for non-monthly timestep'
            #print("nyears is (int(dates_table.shape[0]/12)) :",nyears)
            mb_years=np.arange(nyears)
            #print("mb_years is:",mb_years)

            # Perform inversion
            # - find_inversion_calving_from_any_mb will do the inversion with calving, but if it fails
            #   then it will do the inversion assuming land-terminating
            if invert_standard:
                #print("start do the apparent mb from any mb:")
                #print("invert_standard is True")
                apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears),filesuffix=file_suffix)
                #print("apparent mb from any mb is done")
                tasks.prepare_for_inversion(gdir,filesuffix=file_suffix)
                tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,filesuffix=file_suffix)
            else:
                try:
                    # print("invert_standard is False & The find_inversion_calving_from_any_mb start")
                    # print("⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅")
                    # TODO , Set multiple choices for calving law, if None, it's k_calving, if fa_sermeq_speed_law_inv, then SERMeQ
                    out_calving = find_inversion_calving_from_any_mb(gdir, mb_model= mbmod_inv, mb_years=mb_years,
                                                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,calving_law_inv = None,
                                                                    modelprms = modelprms, glacier_rgi_table = glacier_rgi_table,
                                                                    hindcast=pygem_prms.hindcast,debug=pygem_prms.debug_mb,
                                                                    debug_refreeze=pygem_prms.debug_refreeze,option_areaconstant=True,
                                                                    inversion_filter=False,filesuffix=file_suffix)
                    # print("The find_inversion_calving_from_any_mb end")
                    # print("the out claving is:",out_calving)
                except:
                    print("Something wrong with the find_inversion_calving_from_any_mb")
                    print(traceback.format_exc())
                #print("the calving_flux is",out_calving['calving_flux']   
            # ------ MODEL WITH EVOLVING AREA ------
            # tasks.init_present_time_glacier(gdir,filesuffix=file_suffix) # adds bins below
            geometry_valid = True
            try:
                tasks.init_present_time_glacier(gdir, filesuffix=file_suffix)

            except ValueError as e:
                if "Trapezoid beds need to have origin widths > 0" in str(e):
                    gl_logger.log_invalid_geometry(glacier_str)
                    geometry_valid = False
                else:
                    raise

            if not geometry_valid:
                # Fill the output_df row with safe default values
                # Columns that are scalars
                output_df.loc[nglac, 'calving_k'] = calving_k
                output_df.loc[nglac, 'calving_thick'] = np.nan
                output_df.loc[nglac, 'calving_flux_Gta'] = np.nan
                output_df.loc[nglac, 'no_errors'] = 0
                output_df.loc[nglac, 'oggm_dynamics'] = 0
                # Columns that are lists/arrays (length = nyears)
                list_columns = [
                'length_change_m', 'length_change_rate_myr_dLdt',
                'calving_flux_Gta_timeseries', 'velocity_at_calvingfront_myr',
                'thickness_at_calvingfront_m', 'width_at_calvingfront_m',
                'volume_bsl_m3', 'volume_bwl_m3',
                'frontal_ablation_mwea', 'frontal_ablation_mwea_timeseries',
                'area_km2_timeseries', 'massbal_clim_mwea', 'massbal_total_mwea',
                'massbal_clim_mwea_timeseries', 'massbal_total_mwea_timeseries',
                'massbal_clim_Gta', 'massbal_total_Gta',
                'massbal_clim_Gta_timeseries', 'massbal_total_Gta_timeseries'
                ]
                # Fill row for the failed glacier
                for col in list_columns:
                    if col not in output_df.columns:
                        output_df[col] = np.nan
                    output_df.loc[nglac, col] = np.nan   # SCALAR ONLY
                # Safe scalar outputs for return
                reg_calving_gta_mod_good = 0.0
                reg_calving_gta_obs_good = 0.0

                # Return minimal valid output
                return output_df, reg_calving_gta_mod_good, reg_calving_gta_obs_good, mb_years, mb_obs_mwea, mb_obs_mwea_err

            # Only runs if geometry is valid
            debris.debris_binned(gdir, fl_str='model_flowlines',filesuffix=file_suffix)  # add debris enhancement factors to flowlines
            nfls = gdir.read_pickle('model_flowlines',filesuffix=file_suffix)
            # Mass balance model
            mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                     hindcast=pygem_prms.hindcast,
                                     debug=pygem_prms.debug_mb,
                                     debug_refreeze=pygem_prms.debug_refreeze,
                                     fls=nfls, option_areaconstant=False)
            # Water Level
            # Check that water level is within given bounds
            cls = gdir.read_pickle('inversion_output',filesuffix=file_suffix)[-1]
            th = cls['hgt'][-1]
            thick0 = cls['thick'][-1]
            rho = cfg.PARAMS['ice_density']
            rho_o = cfg.PARAMS['ocean_density'] # Ocean density, must be >= ice density
            water_level = out_calving ['calving_water_level']
            #water_level = -thick0/4 if thick0 > 8*th else 0
            # if gdir.is_tidewater:
            #     if water_level is None:
            #         water_level = -thick0/4 if thick0 > 8*th else 0
            #     else:
            #         water_level = water_level
            if th < (1-rho/rho_o)*thick0:
                print ("Warning: The terminus of this glacier is floating")
                # record in the log file
                #log_floating_warning(glacier_str)  # logs to separate file only
                gl_logger.log_floating_warning(glacier_str)  # logs to separate file only
            #     water_level = th - (1-rho/rho_o)*thick0
            # elif th > 0.3*thick0:
            #     water_level = th - 0.3*thick0
            # else:
            #     water_level = 0
            # vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
            # water_level = utils.clip_scalar(0, th - vmax, th - vmin)

            # print('at the moment water level is :',water_level)
            # print("------------------ after the thickness inversion with calving, run the dynamics ------------------")
            # record the terminus floating status
            record_floating_terminus(glacier_str, th, thick0, water_level, rho, rho_o,index_particles,
                           floating_info_fp_glac=floating_info_fp_glac)


            #%%
            ev_model = CalvingFluxBasedModelJanRt(nfls, y0=0, mb_model=mbmod,
                                      glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                      is_tidewater=gdir.is_tidewater,
                                      water_level=water_level
                                      )
            
            try:
                # print("***********************do the dynamic running with calving***********************")
                # print("nyears is :",nyears)
                try:
                    # add the condition for different situation,1. do_fl_diag = True 2. do_fl_diag = False
                    do_fl_diag = cfg.PARAMS['store_fl_diagnostics']
                    if do_fl_diag:
                        fl_diag_path = gdir.get_filepath('fl_diagnostics',delete=True,filesuffix=file_suffix)
                        diag, fl_diag_dss= ev_model.run_until_and_store(nyears,store_monthly_step= True,fl_diag_path=fl_diag_path)
                        # print('diag is :',diag)
                    else:
                        diag = ev_model.run_until_and_store(nyears,store_monthly_step= True)
                        print('diag is :',diag)
                except:
                    print("something is wrong with the run_until_and_store")
                    print(traceback.format_exc())
                # print("the volume_3 in the diag is :",diag.volume_m3)
                ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
                
                # Record frontal ablation for tidewater glaciers and update total mass balance
                #print("gdir.is_tidewater is:",gdir.is_tidewater)
                #pdb.set_trace()
                if gdir.is_tidewater:
                    try:
                        # Glacier-wide frontal ablation (m3 w.e.)
                        # - note: diag.calving_m3 is cumulative calving
                        #if debug:
                        #    print('\n\ndiag.calving_m3:', diag.calving_m3.values)
                        #    print('calving_m3_since_y0:', ev_model.calving_m3_since_y0)
                        save_path_figure_glac = Path(save_path_figure_glac)
                        save_path_figure_calving = os.path.join(save_path_figure_glac, f"{index_particles}")

                        if Visualize_Index:
                            if not os.path.exists(save_path_figure_calving):
                                os.makedirs(save_path_figure_calving)
                        #print("the calving in the diag is :",diag.calving_m3)
                        # plot the timeseries of calving_m3
                        # print("****************Figure************************")
                        # print("Visualize_Index is:",Visualize_Index)
                        #pdb.set_trace()
                        # if Visualize_Index :
                        #     print("****************Figure************************")
                        #     Visualization_timeseries.plot_timeseries(calving_m3=diag.calving_m3,save_name='Timeseries of accumulated calving flux (m³)',
                        #                                             save_path= save_path_figure_calving)
                        # TODO: Actually calving_m3_monthly here is not annual, it's the timeseries of the model step, monthly/annual, to revise the name later
                        calving_m3_monthly = (diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1]) 
                        #pygem_prms.density_ice / pygem_prms.density_water)
                        # plot the timeseries of calving_m3_monthly
                        # if Visualize_Index :
                        #     Visualization_timeseries.plot_timeseries_Numpy(data = calving_m3_monthly, start_date='2000-01-01', end_date='2019-12-31',
                        #                                                 save_name='Timeseries of calving',save_path=save_path_figure_calving, Y_label='calving flux (m³ a⁻¹)', F_title='Time Series')
                        #print("calving_m3_monthly is:",calving_m3_monthly)
                        # print("the frontalablation is updated totally :",calving_m3_monthly.shape[0])
                        # print(calving_m3_monthly.shape[0],len(ev_model.mb_model.glac_wide_frontalablation))
                        for n in np.arange(calving_m3_monthly.shape[0]):
                            ev_model.mb_model.glac_wide_frontalablation[n] = calving_m3_monthly[n]*pygem_prms.density_ice / pygem_prms.density_water

                        # Glacier-wide climatic and total mass balance (m3 w.e.)

                        ev_model.mb_model.glac_wide_massbaltotal = (
                                ev_model.mb_model.glac_wide_massbaltotal  - ev_model.mb_model.glac_wide_frontalablation)
                        
                        #                    if debug:
                        #                        print('avg calving_m3:', calving_m3_monthly.sum() / nyears)
                        #                        print('avg frontal ablation [Gta]:', 
                        #                              np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                        #                        print('avg frontal ablation [Gta]:', 
                        #                              np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                        #print("model mass balance total is:",ev_model.mb_model.glac_wide_massbaltotal)
                        # ====== Output of calving ======
                        out_calving_forward = {}
                        # area_m2
                        area_m2_monthly = diag.area_m2.values[0:-1]
                        if np.any(area_m2_monthly == 0):
                            raise ValueError("Error: Glacier area (area_m2_monthly) contains zero values, which would result in division by zero.")
                        # area (km2) for the whole period
                        out_calving_forward['area_km2_timeseries'] = area_m2_monthly/ 1e6
                        # mass balance climatic timeseries (m w.e./yr)
                        out_calving_forward['massbal_clim_mwea_timeseries'] = ev_model.mb_model.glac_wide_massbalclim/area_m2_monthly*12
                        # mass balance total timeseries (m w.e./yr)
                        out_calving_forward['massbal_total_mwea_timeseries'] = ev_model.mb_model.glac_wide_massbaltotal/area_m2_monthly*12
                        # mass balance climatic, timeseries (gt /yr)
                        out_calving_forward['massbal_clim_Gta_timeseries'] = ev_model.mb_model.glac_wide_massbalclim * pygem_prms.density_water/1e12 * 12
                        # mass balance total, timeseries (gt /yr)
                        out_calving_forward['massbal_total_Gta_timeseries'] = ev_model.mb_model.glac_wide_massbaltotal * pygem_prms.density_water / 1e12 * 12
                        # mass balance climatic, period-average (m w.e./yr)
                        out_calving_forward['massbal_clim_mwea'] = (ev_model.mb_model.glac_wide_massbalclim/ area_m2_monthly).sum()/nyears
                        # mass balance total, period-average (m w.e./yr)
                        out_calving_forward['massbal_total_mwea'] = (ev_model.mb_model.glac_wide_massbaltotal/ area_m2_monthly).sum()/nyears
                        # mass balance climatic, period-average (Gt/yr)
                        out_calving_forward['massbal_clim_Gta'] = (ev_model.mb_model.glac_wide_massbalclim * pygem_prms.density_water / 1e12).sum()/nyears
                        # mass balance total, period-average (Gt/yr)
                        out_calving_forward['massbal_total_Gta'] = (ev_model.mb_model.glac_wide_massbaltotal * pygem_prms.density_water / 1e12).sum()/nyears
                        # frontal ablation , period-average (m w.e./yr)
                        out_calving_forward['frontal_ablation_mwea'] = (ev_model.mb_model.glac_wide_frontalablation/ area_m2_monthly).sum()/nyears
                        # frontal ablation , timeseries (m w.e./yr)
                        out_calving_forward['frontal_ablation_mwea_timeseries'] = ev_model.mb_model.glac_wide_frontalablation/area_m2_monthly*12

                        # calving flux (km3 ice/yr)
                        out_calving_forward['calving_flux'] = calving_m3_monthly.sum() / nyears / 1e9
                        # calving flux (Gt/yr)
                        #calving_flux_Gta = out_calving_forward['calving_flux'] * pygem_prms.density_ice / pygem_prms.density_water
                        calving_flux_Gta = out_calving_forward['calving_flux'] *1e9* pygem_prms.density_ice / 1e12
                        out_calving_forward['calving_flux_Gta_timeseries'] = calving_m3_monthly*pygem_prms.density_ice/1e12              
                        # calving front thickness at start of simulation
                        #TODO check the last_idx should be the same to the calving law
                        thick = nfls[0].thick
                        last_idx = np.nonzero(thick)[0][-1]
                        out_calving_forward['calving_front_thick'] = thick[last_idx]


                        # Output of length change rate
                        out_calving_forward['length_change_m'] = diag.length_m.values[1:] - diag.length_m.values[0:-1]
                        ## Generate the monthly/annual lenge_change_m and plot timeseries
                        if store_monthly_step:
                            length_change_m_monthly = (diag.length_m.values[1:] - diag.length_m.values[0:-1])
                            length_change_m_annual = np.nansum(length_change_m_monthly.reshape(-1, 12), axis=1)
                            # Visualization_timeseries.plot_timeseries_Numpy(data = length_change_m_monthly, start_date='2000-01-01', end_date='2019-12-31',
                            #                                                save_name='Timeseries of length change',save_path=save_path_figure_calving,
                            #                                                Y_label='length change (m)', F_title='Time Series')
                            # if Visualize_Index:
                            #     Visualization_timeseries.plot_timeseries_List(data = length_change_m_annual, start_year=2000, ylabel= 'length change (m a⁻¹)', xlabel='Year',
                            #                                                 title='Annual timeseries of length change',save_path=save_path_figure_calving,
                            #                                                 save_name='Annual timeseries of length change')
                        else:
                            length_change_m_annual= (diag.length_m.values[1:] - diag.length_m.values[0:-1])
                            # if Visualize_Index:
                            #     Visualization_timeseries.plot_timeseries_List(data = length_change_m_annual, start_year=2000, ylabel= 'length change (m a⁻¹)', xlabel='Year',
                            #                                                 title='Annual timeseries of length change',save_path=save_path_figure_calving,
                            #                                                 save_name='Annual timeseries of length change')
                        
                        ## Generate the monthly/annual lenge_change_rate_myr_dLdt, velocity at the calving front #TODO At the moment, the monthly/annual length change rate and velicity are repeated in the function calib_ind_PBS_MB_FA_RT, which could be optimized later
                        if store_monthly_step:
                            length_change_rate_myr_dLdt_monthly = diag.length_change_rate_myr.values[1:]
                            length_change_rate_myr_dLdt_annual = np.nanmean(length_change_rate_myr_dLdt_monthly.reshape(-1, 12), axis=1)
                            velocity_myr_calvingfront_monthly = diag.velocity_at_calving_front_myr.values[1:]
                            velocity_myr_calvingfront_annual = np.nanmean(velocity_myr_calvingfront_monthly.reshape(-1, 12), axis=1)
                            # Visualization_timeseries.plot_timeseries_Numpy(data = length_change_rate_myr_dLdt_monthly, start_date='2000-01-01', end_date='2019-12-31',
                            #                                                 save_name='Monthly timeseries of length change rate',save_path=save_path_figure_calving,
                            #                                                 Y_label='length change rate (m a⁻¹)', F_title='Timeseries of length change rate (Sermeq)')

                        else:
                            length_change_rate_myr_dLdt_annual = diag.length_change_rate_myr.values[1:]
                            velocity_myr_calvingfront_annual = diag.velocity_at_calving_front_myr.values[1:]

                        # if Visualize_Index:
                            # Visualization_timeseries.plot_timeseries_List(data = length_change_rate_myr_dLdt_annual, start_year=2000, ylabel= 'length change rate (m a⁻¹)', xlabel='Year',
                            #                                                 title='Annual timeseries of length change rate',save_path=save_path_figure_calving,
                            #                                                 save_name='Annual timeseries of length change rate')
                            # Visualization_timeseries.plot_timeseries_List(data = velocity_myr_calvingfront_annual, start_year=2000,
                            #                                                 ylabel='Velocity at the calving front (m a⁻¹)',xlabel= 'Year',
                            #                                                 title='Annual timeseries of velocity at the calving front',save_path=save_path_figure_calving,
                            #                                                 save_name='Annual timeseries of velocity at the calving front')
                        
                        #TODO Check the index, [:-1]should be the start of each time step, and [1:] should be the end of each time step
                        out_calving_forward['length_change_rate_myr_dLdt'] = diag.length_change_rate_myr.values[1:]
                        out_calving_forward['velocity_at_calvingfront_myr'] = diag.velocity_at_calving_front_myr.values[1:]
                        out_calving_forward['thickness_at_calvingfront_m'] = diag.thickness_at_calving_front_m.values[1:]
                        print("********************************************************")
                        print("Available variables in diag:", list(diag.variables))
                        out_calving_forward['width_at_calvingfront_m'] = diag.width_at_calving_front_m.values[1:]
                        out_calving_forward['volume_bsl_m3'] = diag.volume_bsl_m3.values[1:]
                        out_calving_forward['volume_bwl_m3'] = diag.volume_bwl_m3.values[1:]

                        # Plot the timeseries of glacier profile
                        # Visualization_timeseries.plot_timeseries_profile(gdir=gdir, filesuffix ='', save_path=save_path_figure_calving,save_name ='Glacier profile',
                        #                                                 xlabel='Distance along the flowline (m)')
                        
                        #%% Plot the snapshot of each January about the glacier profile
                        # generate selected time list
                        # Define the start and end dates
                        # Visualization_timeseries.plot_time_series_snapshots(gdir=gdir,filesuffix ='', sel_times=None,n_year =1,variable='thickness_m', group='fl_0', 
                        #         ylabel='Elevation (m a.s.l.)', xlabel='Distance along the flowline (m)', title='Time Series Snapshots', 
                        #         save_path=save_path_figure_calving,save_name='Timeseries snapshot of selected year')
                        
                        #%% Plot the animate gif of each month about the glacier profile
                        # if Visualize_Index:
                        #     Visualization_timeseries.animate_time_series(gdir=gdir, filesuffix ='', variable='thickness_m', group='fl_0',interval=400, ylabel='Elevation (m a.s.l.)', 
                        #                                                 xlabel='Distance along the flowline (m)', title='Elevation Changes Animate', save_path=save_path_figure_calving,
                        #                                                 save_name='Animate timeseries of monthly glacier profile')
                        
                        
                        # Record in dataframe
                        # Using apply to set complex data
                        def update_row(row, index, calving_flux_Gta, calving_thick, length_change_m, 
                                       length_change_rate_myr_dLdt,calving_flux_Gta_timeseries,velocity_at_calvingfront_myr,
                                       thickness_at_calvingfront_m,width_at_calvingfront_m,massbal_clim_mwea,massbal_total_mwea,
                                       massbal_clim_mwea_timeseries,massbal_total_mwea_timeseries,volume_bsl_m3,volume_bwl_m3,
                                       frontal_ablation_mwea,frontal_ablation_mwea_timeseries,area_km2_timeseries,massbal_clim_Gta=None,
                                       massbal_total_Gta=None, massbal_clim_Gta_timeseries=None, massbal_total_Gta_timeseries=None):
                            if index == row.name:
                                row['calving_flux_Gta'] = calving_flux_Gta
                                row['calving_thick'] = calving_thick
                                row['no_errors'] = 1
                                row['oggm_dynamics'] = 1
                                row['length_change_m'] = length_change_m
                                row['length_change_rate_myr_dLdt'] = length_change_rate_myr_dLdt
                                row['calving_flux_Gta_timeseries'] = calving_flux_Gta_timeseries
                                row['velocity_at_calvingfront_myr'] = velocity_at_calvingfront_myr
                                row['thickness_at_calvingfront_m'] = thickness_at_calvingfront_m
                                row['width_at_calvingfront_m'] = width_at_calvingfront_m
                                row['massbal_clim_mwea'] = massbal_clim_mwea
                                row['massbal_total_mwea'] = massbal_total_mwea
                                row['massbal_clim_mwea_timeseries'] = massbal_clim_mwea_timeseries
                                row['massbal_total_mwea_timeseries'] = massbal_total_mwea_timeseries
                                row['volume_bsl_m3'] = volume_bsl_m3
                                row['volume_bwl_m3'] = volume_bwl_m3
                                row['frontal_ablation_mwea'] = frontal_ablation_mwea
                                row['frontal_ablation_mwea_timeseries'] = frontal_ablation_mwea_timeseries
                                row['area_km2_timeseries'] = area_km2_timeseries
                                row['massbal_clim_Gta'] = massbal_clim_Gta
                                row['massbal_total_Gta'] = massbal_total_Gta
                                row['massbal_clim_Gta_timeseries'] = massbal_clim_Gta_timeseries
                                row['massbal_total_Gta_timeseries'] = massbal_total_Gta_timeseries
                            return row
                        

                        # Apply the update function to each row
                        output_df = output_df.apply(update_row, axis=1, 
                                                    index=nglac, 
                                                    calving_flux_Gta=calving_flux_Gta,
                                                    calving_thick=out_calving_forward['calving_front_thick'],
                                                    length_change_m=out_calving_forward['length_change_m'].tolist(),  # Convert to list
                                                    length_change_rate_myr_dLdt=out_calving_forward['length_change_rate_myr_dLdt'].tolist(),
                                                    calving_flux_Gta_timeseries=out_calving_forward['calving_flux_Gta_timeseries'].tolist(),
                                                    velocity_at_calvingfront_myr=out_calving_forward['velocity_at_calvingfront_myr'].tolist(),
                                                    thickness_at_calvingfront_m = out_calving_forward['thickness_at_calvingfront_m'].tolist(),
                                                    width_at_calvingfront_m = out_calving_forward['width_at_calvingfront_m'].tolist(),  
                                                    massbal_clim_mwea = out_calving_forward['massbal_clim_mwea'].tolist(),
                                                    massbal_total_mwea = out_calving_forward['massbal_total_mwea'].tolist(),
                                                    massbal_clim_mwea_timeseries = out_calving_forward['massbal_clim_mwea_timeseries'].tolist(),
                                                    massbal_total_mwea_timeseries = out_calving_forward['massbal_total_mwea_timeseries'].tolist(),
                                                    volume_bsl_m3 = out_calving_forward['volume_bsl_m3'].tolist(),
                                                    volume_bwl_m3 = out_calving_forward['volume_bwl_m3'].tolist(),
                                                    frontal_ablation_mwea = out_calving_forward['frontal_ablation_mwea'].tolist(),
                                                    frontal_ablation_mwea_timeseries = out_calving_forward['frontal_ablation_mwea_timeseries'].tolist(),
                                                    area_km2_timeseries = out_calving_forward['area_km2_timeseries'].tolist(),
                                                    massbal_clim_Gta = out_calving_forward.get('massbal_clim_Gta', None),
                                                    massbal_total_Gta = out_calving_forward.get('massbal_total_Gta', None),
                                                    massbal_clim_Gta_timeseries = out_calving_forward.get('massbal_clim_Gta_timeseries', None),
                                                    massbal_total_Gta_timeseries = out_calving_forward.get('massbal_total_Gta_timeseries', None) 
                                                    )
                        
                        
                        # output_df.loc[nglac,'calving_flux_Gta'] = calving_flux_Gta
                        # output_df.loc[nglac,'calving_thick'] = out_calving_forward['calving_front_thick']
                        # output_df.loc[nglac,'no_errors'] = 1
                        # output_df.loc[nglac,'oggm_dynamics'] = 1
                        # print("nglac:", nglac)
                        # print("length_change_m is :",out_calving_forward['length_change_m'])
                        # print("length_change_m length:", len(out_calving_forward['length_change_m']))
                        # output_df.loc[nglac,'length_change_m'] = out_calving_forward['length_change_m']
                        # output_df.loc[nglac,'length_change_rate_myr_dLdt'] = out_calving_forward['length_change_rate_myr_dLdt']
                        
                        if debug:               
                            print('OGGM dynamics + SERMeQ, tau:', np.round(calving_k,4), 'glen_a:', np.round(glen_a_multiplier,2))                 
                            # print('    calving front thickness [m]:', np.round(out_calving_forward['calving_front_thick'],1))
                            # print('    calving flux model multiple-average [Gt/yr]:', np.round(calving_flux_Gta,5))
                            # print('    length change [m] timeseries:', np.round(out_calving_forward['length_change_m'],2))
                            # print('    length change rate [m/yr]:', np.round(out_calving_forward['length_change_rate_myr_dLdt'],2))
                            # print('    calving flux time series [Gt]:', np.round(out_calving_forward['calving_flux_Gta_timeseries'],5))
                            # print('    velocity at calving front [m/yr]:', np.round(out_calving_forward['velocity_at_calvingfront_myr'],2))
                            # print('    thickness at calving front [m]:', np.round(out_calving_forward['thickness_at_calvingfront_m'],2))
                            # print('    width at calving front [m]:', np.round(out_calving_forward['width_at_calvingfront_m'],2))
                            # print('massbalance_climatic model [m w.e. per year]:', np.round(out_calving_forward['massbal_clim_mwea'],5))
                            # print ('massbalance_total model [m w.e. per year]:', np.round(out_calving_forward['massbal_total_mwea'],5))
                            # print('massbalance_climatic model timeseries [m w.e. per year]:', np.round(out_calving_forward['massbal_clim_mwea_timeseries'],5))
                            # print('massbalance_total model timeseries [m w.e. per year]:', np.round(out_calving_forward['massbal_total_mwea_timeseries'],5))
                            # print('volume_bsl [m^3]:', np.round(out_calving_forward['volume_bsl_m3'],2))
                            # print('volume_bwl [m^3]:', np.round(out_calving_forward['volume_bwl_m3'],2))
                            # print('frontal ablation model [m w.e. per year]:', np.round(out_calving_forward['frontal_ablation_mwea'],5))
                            # print('frontal ablation model timeseries [m w.e. per year]:', np.round(out_calving_forward['frontal_ablation_mwea_timeseries'],5))
                    except:
                        print(traceback.format_exc())
                
            except:
                if gdir.is_tidewater:
                    if debug:
                        print('OGGM dynamics failed, using mass redistribution curves')
                        print(traceback.format_exc())
                                                    # Mass redistribution curves glacier dynamics model
                        # #                     ev_model = MassRedistributionCurveModel(
                        # #                                     nfls, mb_model=mbmod, y0=0,
                        # #                                     glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                        # #                                     is_tidewater=gdir.is_tidewater,
                        # #                                     water_level=water_level
                        # #                                     )
                        # #                     _, diag = ev_model.run_until_and_store(nyears)
                        # #                     ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                        # #                     ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values
                            
                        # #                     # Record frontal ablation for tidewater glaciers and update total mass balance
                        # #                     # Update glacier-wide frontal ablation (m3 w.e.)
                        # #                     ev_model.mb_model.glac_wide_frontalablation = ev_model.mb_model.glac_bin_frontalablation.sum(0)
                        # #                     # Update glacier-wide total mass balance (m3 w.e.)
                        # #                     ev_model.mb_model.glac_wide_massbaltotal = (
                        # #                             ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation)

                        # #                     calving_flux_km3a = (ev_model.mb_model.glac_wide_frontalablation.sum() * pygem_prms.density_water / 
                        # #                                          pygem_prms.density_ice / nyears / 1e9)

                        # # #                    if debug:
                        # # #                        print('avg frontal ablation [Gta]:', 
                        # # #                              np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
                        # # #                        print('avg frontal ablation [Gta]:', 
                        # # #                              np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                                            
                        # #                     # Output of calving
                        # #                     out_calving_forward = {}
                        # #                     # calving flux (km3 ice/yr)
                        # #                     out_calving_forward['calving_flux'] = calving_flux_km3a
                        # #                     # calving flux (Gt/yr)
                        # #                     calving_flux_Gta = out_calving_forward['calving_flux'] * pygem_prms.density_ice / pygem_prms.density_water
                        # #                     # calving front thickness at start of simulation
                        # #                     thick = nfls[0].thick
                        # #                     last_idx = np.nonzero(thick)[0][-1]
                        # #                     out_calving_forward['calving_front_thick'] = thick[last_idx]
                                            
                        # #                     # Record in dataframe
                        # #                     output_df.loc[nglac,'calving_flux_Gta'] = calving_flux_Gta
                        # #                     output_df.loc[nglac,'calving_thick'] = out_calving_forward['calving_front_thick']
                        # #                     output_df.loc[nglac,'no_errors'] = 1
                                            
                        # #                     if debug:          
                        # #                         print('Mass Redistribution curve, calving_k:', np.round(calving_k,1), 'glen_a:', np.round(glen_a_multiplier,2))                       
                        # #                         print('    calving front thickness [m]:', np.round(out_calving_forward['calving_front_thick'],0))
                        # #                         print('    calving flux model [Gt/yr]:', np.round(calving_flux_Gta,5))


            if calc_mb_geo_correction:
                # Mass balance correction from mass loss above sea level due to calving retreat 
                #  (i.e., what the geodetic signal should see)
                last_yr_idx = np.where(mbmod.glac_wide_area_annual > 0)[0][-1]
                if last_yr_idx == mbmod.glac_bin_area_annual.shape[1]-1:
                    last_yr_idx = -2
                bin_last_idx = np.where(mbmod.glac_bin_area_annual[:,last_yr_idx] > 0)[0][-1]
                bin_area_lost = mbmod.glac_bin_area_annual[bin_last_idx:,0] - mbmod.glac_bin_area_annual[bin_last_idx:,-2]
                height_asl = mbmod.heights - water_level
                height_asl[mbmod.heights<0] = 0
                mb_mwea_fa_asl_geo_correction = ((bin_area_lost * height_asl[bin_last_idx:]).sum() / 
                                        mbmod.glac_wide_area_annual[0] *
                                        pygem_prms.density_ice / pygem_prms.density_water / nyears)
                mb_mwea_fa_asl_geo_correction_max = 0.3*gta_to_mwea(calving_flux_Gta, glacier_rgi_table['Area']*1e6)
                if mb_mwea_fa_asl_geo_correction > mb_mwea_fa_asl_geo_correction_max:
                    mb_mwea_fa_asl_geo_correction = mb_mwea_fa_asl_geo_correction_max
                    
                # Below sea-level correction due to calving that geodetic mass balance doesn't see
#                print('test:', mbmod.glac_bin_icethickness_annual.shape, height_asl.shape, bin_area_lost.shape)
#                height_bsl = mbmod.glac_bin_icethickness_annual - height_asl
                
                # Area for retreat
                if debug:
#                    print('\n----- area calcs -----')
#                    print(mbmod.glac_bin_area_annual[bin_last_idx:,0])
#                    print(mbmod.glac_bin_icethickness_annual[bin_last_idx:,0])
#                    print(mbmod.glac_bin_area_annual[bin_last_idx:,-2])
#                    print(mbmod.glac_bin_icethickness_annual[bin_last_idx:,-2])
#                    print(mbmod.heights.shape, mbmod.heights[bin_last_idx:])
                    print('  mb_mwea_fa_asl_geo_correction:', np.round(mb_mwea_fa_asl_geo_correction,2))
#                    print('  mb_mwea_fa_asl_geo_correction:', mb_mwea_fa_asl_geo_correction)
#                    print(glacier_rgi_table, glacier_rgi_table['Area'])
                    
                    
                output_df.loc[nglac,'mb_mwea_fa_asl_lost'] = mb_mwea_fa_asl_geo_correction
            #
            # print("out_calving_forward is :",out_calving_forward)

            if out_calving_forward is None:
                output_df.loc[nglac,['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors','length_change_m',
                                     'length_change_rate_myr_dLdt','calving_flux_Gta_timeseries','velocity_at_calvingfront_myr',
                                     'thickness_at_calvingfront_m','width_at_calvingfront_m','massbal_clim_mwea','massbal_total_mwea',
                                     'massbal_clim_mwea_timeseries','massbal_total_mwea_timeseries','volume_bsl_m3','volume_bwl_m3',
                                     'frontal_ablation_mwea','frontal_ablation_mwea_timeseries','area_km2_timeseries',
                                     'massbal_clim_Gta','massbal_total_Gta','massbal_clim_Gta_timeseries',
                                     'massbal_total_Gta_timeseries']] = (np.nan, np.nan, np.nan, 0,np.nan,np.nan,np.nan,np.nan,np.nan,
                                                                         np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                                                                         np.nan,np.nan,np.nan,np.nan,np.nan)
                
    # Remove glaciers that failed to run
    if fa_glac_data_reg is None:
        reg_calving_gta_obs_good = None
        output_df_good = output_df.dropna(axis=0, subset=['calving_flux_Gta'])
        reg_calving_gta_mod_good = output_df_good.calving_flux_Gta.sum()
    elif ignore_nan:
        output_df_good = output_df.dropna(axis=0, subset=['calving_flux_Gta'])
        reg_calving_gta_mod_good = output_df_good.calving_flux_Gta.sum()
        rgiids_data = list(fa_glac_data_reg.RGIId.values)
        rgiids_mod = list(output_df_good.RGIId.values)
        fa_data_idx = [rgiids_data.index(x) for x in rgiids_mod]
        reg_calving_gta_obs_good = fa_glac_data_reg.loc[fa_data_idx,'fa_gta_obs'].sum()
    else:
        reg_calving_gta_mod_good = output_df.calving_flux_Gta.sum()
        reg_calving_gta_obs_good = fa_glac_data_reg['fa_gta_obs'].sum()
            
    return output_df, reg_calving_gta_mod_good, reg_calving_gta_obs_good,mb_years,mb_obs_mwea,mb_obs_mwea_err


# the function for paralle running
def processing_parameters(model_function,kwargs,modelprms_MB_FA):
    """Process model parameters with configurable logging
    
    Args:
        model_function: The model function to execute
        kwargs: Dictionary of keyword arguments including logging parameters
        modelprms_MB_FA: Model parameters to process
        
    Returns:
        dict: Dictionary containing all model outputs
        
    Raises:
        Exception: Propagates any exceptions from model execution with logging
    """
    log_level = kwargs.get('log_level', 'INFO')  # fallback to 'INFO' if missing
    # Initialize logger at the start (if log_level is DEBUG)
    logger = None
    if log_level == 'DEBUG':
        logger = setup_worker_logger(
            save_path_log=save_path_log,
            log_level=log_level
        )
        logger.debug(f"Processing modelprms_MB_FA={modelprms_MB_FA}")
    
    try:
        output_df, reg_calving_gta_mod_good, _, mb_years, mb_obs_mwea, mb_obs_mwea_err = (
            model_function(
                modelprms_MB_FA=modelprms_MB_FA,
                do_DA_calib_Paralle=True,
                **kwargs
            )
        )

        out_dict = {
            'modelprms_MB_FA_value': modelprms_MB_FA,
            'calving_gta_average_regionalsum': reg_calving_gta_mod_good,
            'length_change_m_timeseries': output_df['length_change_m'].tolist(),
            'length_change_rate_myr_dLdt': output_df['length_change_rate_myr_dLdt'].tolist(),
            'calving_thick': output_df['calving_thick'].tolist(),
            'calving_flux_Gta_average': output_df['calving_flux_Gta'].tolist(),
            'calving_flux_Gta_timeseries': output_df['calving_flux_Gta_timeseries'].tolist(),
            'massbal_clim_mwea': output_df['massbal_clim_mwea'].tolist(),
            'massbal_total_mwea': output_df['massbal_total_mwea'].tolist(),
            'massbal_clim_Gta': output_df['massbal_clim_Gta'].tolist(),
            'massbal_total_Gta': output_df['massbal_total_Gta'].tolist(),
            'massbal_clim_mwea_timeseries': output_df['massbal_clim_mwea_timeseries'].tolist(),
            'massbal_total_mwea_timeseries': output_df['massbal_total_mwea_timeseries'].tolist(),
            'massbal_clim_Gta_timeseries': output_df['massbal_clim_Gta_timeseries'].tolist(),
            'massbal_total_Gta_timeseries': output_df['massbal_total_Gta_timeseries'].tolist(),
            'velocity_at_calvingfront_myr': output_df['velocity_at_calvingfront_myr'].tolist(),
            'thickness_at_calvingfront_m': output_df['thickness_at_calvingfront_m'].tolist(),
            'width_at_calvingfront_m': output_df['width_at_calvingfront_m'].tolist(),
            'volume_bsl_m3': output_df['volume_bsl_m3'].tolist(),
            'volume_bwl_m3': output_df['volume_bwl_m3'].tolist(),
            'frontal_ablation_mwea': output_df['frontal_ablation_mwea'].tolist(),
            'frontal_ablation_mwea_timeseries': output_df['frontal_ablation_mwea_timeseries'].tolist(),
            'area_km2_timeseries': output_df['area_km2_timeseries'].tolist(),
            'mb_years': mb_years,
            'mb_obs_mwea': mb_obs_mwea,
            'mb_obs_mwea_err': mb_obs_mwea_err,
        }

        return {
            "success": True,
            "data": out_dict,
            "error": None,
            "traceback": None,
        }

    except Exception as e:
        if logger:
            logger.error(f"Failure for {modelprms_MB_FA}: {e}")
            logger.exception("Traceback")
        return {
            "success": False,
            "data": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # try:
    #     if logger:   
    #         logger.info(f"Processing sample: {modelprms_MB_FA}")
        
    #     output_df, reg_calving_gta_mod_good,_ , mb_years,mb_obs_mwea,mb_obs_mwea_err= model_function(modelprms_MB_FA = modelprms_MB_FA, 
    #                                                                                                  do_DA_calib_Paralle = True, **kwargs)
        
    #     # Convert dates_table (DataFrame) to a structured NumPy array
    #     mb_years_np = mb_years
    #     # Create the output dictionary
    #     out_dict =  {
    #                 'modelprms_MB_FA_value' : modelprms_MB_FA,
    #                 'calving_gta_average_regionalsum': reg_calving_gta_mod_good,
    #                 'length_change_m_timeseries': output_df['length_change_m'].tolist(),
    #                 'length_change_rate_myr_dLdt': output_df['length_change_rate_myr_dLdt'].tolist(),
    #                 'calving_thick': output_df['calving_thick'].tolist(),
    #                 'calving_flux_Gta_average': output_df['calving_flux_Gta'].tolist(),
    #                 'calving_flux_Gta_timeseries':output_df['calving_flux_Gta_timeseries'].tolist(),
    #                 'massbal_clim_mwea': output_df['massbal_clim_mwea'].tolist(),
    #                 'massbal_total_mwea': output_df['massbal_total_mwea'].tolist(),
    #                 'massbal_clim_Gta': output_df['massbal_clim_Gta'].tolist(),
    #                 'massbal_total_Gta': output_df['massbal_total_Gta'].tolist(),
    #                 'massbal_clim_mwea_timeseries': output_df['massbal_clim_mwea_timeseries'].tolist(),
    #                 'massbal_total_mwea_timeseries': output_df['massbal_total_mwea_timeseries'].tolist(),
    #                 'massbal_clim_Gta_timeseries': output_df['massbal_clim_Gta_timeseries'].tolist(),
    #                 'massbal_total_Gta_timeseries': output_df['massbal_total_Gta_timeseries'].tolist(),
    #                 'velocity_at_calvingfront_myr': output_df['velocity_at_calvingfront_myr'].tolist(),
    #                 'thickness_at_calvingfront_m': output_df['thickness_at_calvingfront_m'].tolist(),
    #                 'width_at_calvingfront_m': output_df['width_at_calvingfront_m'].tolist(),
    #                 'volume_bsl_m3': output_df['volume_bsl_m3'].tolist(),
    #                 'volume_bwl_m3': output_df['volume_bwl_m3'].tolist(),
    #                 'frontal_ablation_mwea': output_df['frontal_ablation_mwea'].tolist(),
    #                 'frontal_ablation_mwea_timeseries': output_df['frontal_ablation_mwea_timeseries'].tolist(),
    #                 'area_km2_timeseries': output_df['area_km2_timeseries'].tolist(),
    #                 'mb_years': mb_years_np,  # Add structured NumPy array of dates_table,
    #                 'mb_obs_mwea': mb_obs_mwea,
    #                 'mb_obs_mwea_err': mb_obs_mwea_err
    #                 }
    #     if logger:
    #         logger.info(f"Completed sample: {modelprms_MB_FA}")
    #     return out_dict
    # except Exception as e:
    #     error_msg = f"Error processing sample {modelprms_MB_FA}: {str(e)}"
    #     if logger:
    #         logger.error(error_msg)
    #         logger.exception("Full error traceback:")
    #     else:
    #         print(error_msg)
    #         print(f"Error type: {type(e).__name__}")
    #     raise  # Re-raise the exception after logging


def Visualize_parameter_paralle (model_function = None, parameters_dict = None,calibrate_timeseries = False,
                                 rgiid_ind = None, Visual_index = False, N_iteration = None,
                                 store_result =False,save_path_figure_glac = None,
                                 save_path_parameter_glac =None,save_path_modeloutput_glac=None,
                                 save_path_log_glac = None,log_level = None,floating_info_fp_glac=None, **kwargs):
    """
        This function is used to visulize the relationship between parameter k and model_functions, copy from def Visualize_parameter, revised for
    paralle computing 

    Args:
        model_function: the function to be calibrated, the function should return the output of the model
        parameters_dict: the dictionary of the parameters, the key is the name of the parameter, the value is the list of the parameter
        calibrate_timeseries: boolean, if True, the function is used to calibrate the annual timeseries of length change, vice verse;
        The defaule is False, just calibrate the multiple year averaged FA
        rgiid_ind: the glacier id
        Visual_index: the boolean, if True, the function will visualize the relationship between parameters and output of the glacier
        N_iteration: int the number of the iteration or str ("poster") for the poster
        store_result: boolean, if True, the function will store the result in the output folder
        save_path_figure_glac: str, the path of the figure folder for single glacier
        save_path_parameter_glac: str, the path of the parameter folder for single glacier
        save_path_modeloutput_glac: str, the path of the model output folder for single glacier
        save_path_log_glac: str, the path of the log folder for single glacier
        log_level: str, the log level for the logger,the default is 'INFO', could be 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL', also control the print hints
        floating_info_fp_glac: str, the path to the floating info file for the glacier
        **kwargs: other arguments for the model_function
    Returns:
        The model ouput of the paralle computing for the model_function with the parameters_dict


    # the prior modelprms_MB_FA is the parameter of the model_function (Tbias,kb,ddfsnow,tau,Index) 
    #Sample_N = pygem_prms.pbs_sample_no
    #prior_samples = sample_prior(Sample_N)
    """
    N_sample = len(parameters_dict['tbias'])
    prior_samples = parameters_dict
    prior_samples_list = [{key: prior_samples[key][i] for key in prior_samples} for i in range(len(prior_samples['tbias']))]

    reg_calving_gta_mod_good = np.zeros(N_sample)


    # parallel compute the model output
    proc_count = cpu_count()
    print(f"There are {proc_count} processors are available")
    proc_count_RT = max(1, proc_count - proc_count//2)  # Ensure at least one process
    print(f"Using {proc_count_RT} processors")

    if calibrate_timeseries:
        try:
        # Use Pool with partial for additional arguments
            # Main process logging setup
            main_logger = logging.getLogger('main')
            main_logger.setLevel(log_level)
            
            # Proper path joining using pathlib
            main_log_path = Path(save_path_log_glac) / 'main_process.log'
            main_handler = logging.FileHandler(main_log_path)
            
            # Clear existing handlers to avoid duplicates
            main_logger.handlers = []
            main_logger.addHandler(main_handler)
            
            # Formatting
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            main_handler.setFormatter(formatter)
            
            main_logger.info("Starting parallel processing...")
            kwargs['save_path_figure_glac'] = save_path_figure_glac
            # kwargs['save_path_parameter_glac'] = save_path_parameter_glac
            # kwargs['save_path_modeloutput_glac'] = save_path_modeloutput_glac
            # kwargs['save_path_log_glac'] = save_path_log_glac
            kwargs['log_level'] = log_level
            process_func = partial(processing_parameters, model_function,kwargs)

            with Pool(proc_count_RT) as pool:
                output = pool.map(process_func, prior_samples_list)
                # ----------------------------------------
                # Unwrap parallel outputs (NEW, REQUIRED)
                # ----------------------------------------
                success_flags = np.array([res["success"] for res in output])

                if not success_flags.any():
                    raise RuntimeError(
                        f"All parameter samples failed for glacier {rgiid_ind}"
                    )

                # Log failures but do NOT crash
                n_fail = (~success_flags).sum()
                if n_fail > 0:
                    print(
                        f"[WARNING] {n_fail}/{len(output)} samples failed "
                        f"for glacier {rgiid_ind}"
                    )

                # Extract only successful data dicts
                output_success = [res["data"] for res in output if res["success"]]
                
            # Ensure output is not empty
            if not output_success:
                raise ValueError("Processing function returned an empty output. Check 'process_func' or 'prior_samples'.")
            # Extract the results
            # Extract results using dictionary comprehension
            keys = [
                'modelprms_MB_FA_value','calving_gta_average_regionalsum', 'length_change_m_timeseries','length_change_rate_myr_dLdt',
                'calving_thick', 'calving_flux_Gta_average','calving_flux_Gta_timeseries', 'massbal_clim_mwea',
                'massbal_total_mwea', 'massbal_clim_mwea_timeseries', 'massbal_total_mwea_timeseries',
                'massbal_clim_Gta', 'massbal_total_Gta', 'massbal_clim_Gta_timeseries','massbal_total_Gta_timeseries',
                'velocity_at_calvingfront_myr','thickness_at_calvingfront_m','width_at_calvingfront_m','volume_bsl_m3','volume_bwl_m3',
                'frontal_ablation_mwea', 'frontal_ablation_mwea_timeseries','area_km2_timeseries','mb_years', 'mb_obs_mwea','mb_obs_mwea_err'] # TODO ADD THE 'dates_table' to the keys, 'modelprms_MB_FA_value'
            # Convert extracted values to NumPy arrays
            #output_data = {key: np.array([out_dict[key] for out_dict in output]) for key in keys}
            output_data = {
                            key: np.asarray(
                                [out[key] for out in output_success],
                                dtype=object
                            )
                            for key in keys
                        }
            if log_level == 'DEBUG':
                print("output is :",output)
                print("output_data is:",output_data)
            # Convert modelprms_MB_FA_value separately
            modelprms_data = output_data['modelprms_MB_FA_value']
            modelprms_data_serializable = [
                {key: (value.item() if isinstance(value, (np.generic, np.ndarray)) else value) for key, value in item.items()}
                for item in modelprms_data
            ]

            # Handle structured arrays like 'dates_table'
            # if 'dates_table' in output_data:
            #     dates_table_serializable = [
            #         {name: row[name].item() if isinstance(row[name], np.generic) else row[name] for name in output_data['dates_table'].dtype.names}
            #         for row in output_data['dates_table']
            #     ]
            #     output_data['dates_table'] = dates_table_serializable

            # Convert other NumPy arrays and handle NumPy-specific types
            output_data_serializable = {
                key: (
                    value.tolist() if isinstance(value, np.ndarray) else 
                    [{k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in row.items()} for row in value] 
                    if isinstance(value, list) and isinstance(value[0], dict) else 
                    (value.item() if isinstance(value, np.generic) else value)
                )
                for key, value in output_data.items() if key != 'modelprms_MB_FA_value'
            }

            #output_data_serializable = {key: value.tolist() for key, value in output_data.items()}

            modelprms_MB_FA = output_data['modelprms_MB_FA_value']
            reg_calving_gta_mod_good = output_data['calving_gta_average_regionalsum']
            lengthchange_m_TMS = output_data['length_change_m_timeseries']
            assert isinstance(output_data['length_change_m_timeseries'][0], (list, np.ndarray))
            lengthchange_rate_dLdt = output_data['length_change_rate_myr_dLdt']
            calving_thickness_model = output_data['calving_thick']
            calving_flux_Gta_average = output_data['calving_flux_Gta_average']
            calving_flux_Gta_TMS = output_data['calving_flux_Gta_timeseries']
            massbal_clim = output_data['massbal_clim_mwea']
            massbal_total = output_data['massbal_total_mwea']
            massbal_clim_timeseries = output_data['massbal_clim_mwea_timeseries']
            massbal_total_timeseries = output_data['massbal_total_mwea_timeseries']
            massbal_clim_Gta = output_data['massbal_clim_Gta']
            massbal_total_Gta = output_data['massbal_total_Gta']
            massbal_clim_Gta_timeseries = output_data['massbal_clim_Gta_timeseries']
            massbal_total_Gta_timeseries = output_data['massbal_total_Gta_timeseries']
            velocity_at_calvingfront = output_data['velocity_at_calvingfront_myr']
            thickness_at_calvingfront = output_data['thickness_at_calvingfront_m']
            width_at_calvingfront = output_data['width_at_calvingfront_m']
            volume_bsl = output_data['volume_bsl_m3']
            volume_bwl = output_data['volume_bwl_m3']
            frontal_ablation = output_data['frontal_ablation_mwea']
            frontal_ablation_timeseries = output_data['frontal_ablation_mwea_timeseries']
            area_km2_timeseries = output_data['area_km2_timeseries']
            mb_years = output_data['mb_years']
        except Exception as e:
            print(f"An error occurred during parallel processing: {e}")
            print(traceback.format_exc())
    else:
        reg_calving_gta_mod_good = []
        for i, modelprms_mb_fa in enumerate(modelprms_MB_FA):
            _, reg_calving_gta_mod_good[i],_ = model_function(modelprms_MB_FA = modelprms_mb_fa, **kwargs)
    
    # Visualize the relationship and save the figure
    if Visual_index:
        data_visulization = {
            'Tbias': prior_samples['tbias'],
            'kp': prior_samples['kp'],
            'ddfsnow': prior_samples['ddfsnow'],
            'tau': prior_samples['tau'],
            'calving_gta_average': reg_calving_gta_mod_good,
            'frontal_ablation_mwea': frontal_ablation,
            'dLdt': lengthchange_rate_dLdt,
            'massbal_clim': massbal_clim
        }
        
        # Call the plot function
        #TODO add the function plot_and_save_relationship in Visulization_timeseries
        Visualization_timeseries.plot_and_save_relationship(data =data_visulization, parameters = ['Tbias','kp','ddfsnow','tau'], result = 'calving_gta_average', save_path =save_path_figure_glac)
        Visualization_timeseries.plot_and_save_relationship(data =data_visulization, parameters = ['Tbias','kp','ddfsnow','tau'], result = 'frontal_ablation_mwea', save_path =save_path_figure_glac)
        Visualization_timeseries.plot_and_save_relationship(data =data_visulization, parameters = ['Tbias','kp','ddfsnow','tau'], result = 'massbal_clim', save_path =save_path_figure_glac)

    # Debug output
    if log_level == 'DEBUG':
        print("Parameter_values:", prior_samples)
        print("reg_calving_gta_mod_good:", reg_calving_gta_mod_good)
        print("lengthchange_rate_dLdt:", lengthchange_rate_dLdt)
        print("the type of lengthchange_rate_dLdt is :",type(lengthchange_rate_dLdt))
        print("the calving flux Gt is :",calving_flux_Gta_TMS)
        print("the velocity at the calvingfront myr is :",velocity_at_calvingfront)
        print(lengthchange_rate_dLdt.apply(type))

    # remove the output gdir directory of the current glacier with the specific tau
    rgiid_ind_float = f"{float(rgiid_ind.split('-')[1]):0.5f}"
    index_particles = prior_samples['index']
    for k in index_particles:
        #os.rmdir(pygem_prms.oggm_gdir_fp + str(k))
        shutil.rmtree(pygem_prms.oggm_gdir_fp + rgiid_ind_float+'_'+f"{k:.0f}" )  # Use rmtree to remove non-empty directories
    # for folder in glob.glob(os.path.join(pygem_prms.oggm_gdir_fp, rgiid_ind_float + '_*')):
    #     shutil.rmtree(folder, ignore_errors=True)
    #plt.show()
    if calibrate_timeseries:
        # save the output as hpf5 file
        # Define output file path
        # output_folder_model = os.path.join(output_fp, 'modeloutput')  # Assuming `pygem_prms.output_fp` exists
        # output_folder_params = os.path.join(output_fp, 'parameter')
        output_folder_model = os.path.join(save_path_modeloutput_glac,'Prior') 
        output_folder_params =os.path.join(save_path_parameter_glac,'Prior')  
        # Ensure directories exist
        os.makedirs(output_folder_model, exist_ok=True)
        os.makedirs(output_folder_params, exist_ok=True)
        output_filename = f'calibration_prior_model_Original_output_{rgiid_ind}_{N_iteration}.json'
        output_filename_params = f'calibration_prior_Params_{rgiid_ind}_{N_iteration}.json'
        output_fp_prior = os.path.join(output_folder_model, output_filename)
        output_fp_params = os.path.join(output_folder_params, output_filename_params)

        if store_result:
        # # Save to JSON
            try:
                # Save modelprms_MB_FA_value to JSON
                #pdb.set_trace()
                with open(output_fp_params, "w") as f:
                    json.dump(modelprms_data_serializable, f, indent=4)
                
                with open(output_fp_prior, "w") as f:
                    json.dump(output_data_serializable, f, indent=4)
            except:
                print(f"Error saving output to {output_fp_prior}")
                print(traceback.format_exc())
            
            #pdb.set_trace()
            prior_samples_serializable = {key: value.tolist() for key, value in prior_samples.items()}

            with open(f"{output_folder_params}/prior_params_samples_{rgiid_ind}_{N_iteration}.json", "w") as f:
                json.dump(prior_samples_serializable , f, indent=4)
    

        #return prior_samples, output_data
        return output_data
    else:
        #return prior_samples, reg_calving_gta_mod_good
        return reg_calving_gta_mod_good
    


def Model_MB_FA_RT(model_function = reg_calving_flux,parameters_dict= None,rgiid_ind = None,
                   main_glac_rgi = None, fa_glac_data_reg= None,ignore_nan=False,
                   calibrate_timeseries =True,store_monthly_step=False, return_all = False,
                   store_result= False,N_iteration = None,save_path_figure_glac = None,
                   save_path_parameter_glac = None,save_path_modeloutput_glac = None,
                   save_path_log_glac=None,log_level = None,floating_info_fp_glac=None, **kwargs):
    """
    This function is the main function for the  computation of the mass balance and Frontal ablation/lengthchange, is the coupling of PyGEM, OGGM, SERMeQ

    Parameters
    ----------
    model_function : callable, optional
    parameters_dict : dict, optional
    rgi_ind : str, optional
    main_glac_rgi : str, optional
    fa_glac_data_reg : str, optional
    ignore_nan : bool, optional
    calibrate_timeseries : bool, optional
    store_monthly_step : bool, optional
    return_all : bool, optional, if True, the function will return all the output of the model_function, the default is False, 
        just returned the output for the calibration
    N_iteration : srt , str(int), optional, the number of the iterations for AMIS In the calibration
    save_path_figure_glac : str, optional, the path to store the figures
    save_path_parameter_glac : str, optional, the path to store the parameters
    save_path_modeloutput_glac : str, optional, the path to store the model output
    log_level : str, optional, the log level for the logger, the default is None, which means the log level is INFO, could be DEBUG, WARNING, ERROR, CRITICAL
    save_path_log_glac : str, optional, the path to store the log file
    floating_info_fp_glac : str, optional, the path to the floating info file for the glacier
    kwargs : dict, optional
    Returns
    -------
    """
    output_prior =  Visualize_parameter_paralle (model_function = model_function,parameters_dict= parameters_dict,rgiid_ind = rgiid_ind,
                                                                    main_glac_rgi = main_glac_rgi, fa_glac_data_reg=fa_glac_data_reg,ignore_nan=ignore_nan,
                                                                    calibrate_timeseries =calibrate_timeseries,store_monthly_step=store_monthly_step,
                                                                    N_iteration=N_iteration,store_result=store_result,save_path_figure_glac=save_path_figure_glac,
                                                                    save_path_parameter_glac =save_path_parameter_glac,save_path_modeloutput_glac=save_path_modeloutput_glac,
                                                                    save_path_log_glac=save_path_log_glac,log_level = log_level,floating_info_fp_glac=floating_info_fp_glac, **kwargs)
    # Extract the results
    # Extract results using dictionary comprehension (#TODOat the moment, we using lengthchnage_dLdt_model_array, and massbalclim_model_array to do the calibration, more choice could be added in the future)
    lengthchange_m_TMS_model_array = output_prior['length_change_m_timeseries'] # lengthchange based on the difference of the length of the centerline flowline
    lengthchange_dLdt_model_array = output_prior['length_change_rate_myr_dLdt'] # lengthchange based on the calving law (SERMeQ)
    calving_flux_Gta_TMS_model_array = output_prior['calving_flux_Gta_timeseries']
    calving_flux_Gta_average_model_array = output_prior['calving_flux_Gta_average']
    massbalclim_model_array = output_prior['massbal_clim_mwea']
    massbaltotal_model_array = output_prior['massbal_total_mwea']
    massbalclim_Gta_model_array = output_prior['massbal_clim_Gta']
    massbaltotal_Gta_model_array = output_prior['massbal_total_Gta']
    massbalclim_TMS_model_array = output_prior['massbal_clim_mwea_timeseries']
    massbaltotal_TMS_model_array = output_prior['massbal_total_mwea_timeseries']
    massbalclim_Gta_TMS_model_array = output_prior['massbal_clim_Gta_timeseries']
    massbaltotal_Gta_TMS_model_array = output_prior['massbal_total_Gta_timeseries']
    FA_mwea_average_model_array = output_prior['frontal_ablation_mwea']
    FA_mwea_TMS_model_array = output_prior['frontal_ablation_mwea_timeseries']
    area_km2_TMS_model_array = output_prior['area_km2_timeseries']
    velocity_at_calvingfront_model_array = output_prior['velocity_at_calvingfront_myr']
    thickness_at_calvingfront_model_array = output_prior['thickness_at_calvingfront_m']
    width_at_calvingfront_model_array = output_prior['width_at_calvingfront_m']
    volume_bsl_model_array = output_prior['volume_bsl_m3']
    volume_bwl_model_array = output_prior['volume_bwl_m3']
    calving_thickness_model_array = output_prior['calving_thick']
    mb_obs_mwea = output_prior['mb_obs_mwea'][0] #TODO chekck the mass balance data should be the climatic mass balance
    mb_obs_mwea_err = output_prior['mb_obs_mwea_err'][0] # [0], the observation are the same for all the particles
    #pdb.set_trace() 
    #output_folder_model  = os.path.join(save_path_modeloutput, glacier_str.split('.')[0].zfill(2))
    if store_monthly_step:
        # Flatten the nested lists
        lengthchange_m_TMS_model_array_flattened = [sublist[0] for sublist in lengthchange_m_TMS_model_array]
        lengthchange_m_TMS_model_array_monthly =  lengthchange_m_TMS_model_array_flattened
        lengthchange_m_TMS_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in lengthchange_m_TMS_model_array_monthly]
        
        lengthchange_dLdt_model_array_flattened = [sublist[0] for sublist in lengthchange_dLdt_model_array]
        lengthchange_dLdt_model_array_monthly =  lengthchange_dLdt_model_array_flattened
        lengthchange_dLdt_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in lengthchange_dLdt_model_array_monthly]
        
        calving_flux_Gta_TMS_model_array_flattened = [sublist[0] for sublist in calving_flux_Gta_TMS_model_array]
        calving_flux_Gta_TMS_model_array_monthly =  calving_flux_Gta_TMS_model_array_flattened
        calving_flux_Gta_TMS_model_array_annual = [[np.sum(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in calving_flux_Gta_TMS_model_array_monthly]
        
        massbalclim_TMS_model_array_flattened = [sublist[0] for sublist in massbalclim_TMS_model_array]
        massbalclim_TMS_model_array_monthly =  massbalclim_TMS_model_array_flattened
        massbalclim_TMS_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in massbalclim_TMS_model_array_monthly]
        
        massbaltotal_TMS_model_array_flattened = [sublist[0] for sublist in massbaltotal_TMS_model_array]
        massbaltotal_TMS_model_array_monthly =  massbaltotal_TMS_model_array_flattened
        massbaltotal_TMS_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in massbaltotal_TMS_model_array_monthly]

        massbalclim_Gta_TMS_model_array_flattened = [sublist[0] for sublist in massbalclim_Gta_TMS_model_array]
        massbalclim_TMS_model_array_monthly_gta =  massbalclim_Gta_TMS_model_array_flattened
        massbalclim_TMS_model_array_annual_gta = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in massbalclim_TMS_model_array_monthly_gta]

        massbaltotal_Gta_TMS_model_array_flattened = [sublist[0] for sublist in massbaltotal_Gta_TMS_model_array]
        massbaltotal_TMS_model_array_monthly_gta =  massbaltotal_Gta_TMS_model_array_flattened
        massbaltotal_TMS_model_array_annual_gta = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in massbaltotal_TMS_model_array_monthly_gta]

        FA_mwea_TMS_model_array_flattened = [sublist[0] for sublist in FA_mwea_TMS_model_array]
        FA_mwea_TMS_model_array_monthly =  FA_mwea_TMS_model_array_flattened
        FA_mwea_TMS_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in FA_mwea_TMS_model_array_monthly]

        area_km2_TMS_model_array_flattened = [sublist[0] for sublist in area_km2_TMS_model_array]
        area_km2_TMS_model_array_monthly =  area_km2_TMS_model_array_flattened
        area_km2_TMS_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in area_km2_TMS_model_array_monthly]

        #pdb.set_trace()
        velocity_at_calvingfront_model_array_flattened = [sublist[0] for sublist in velocity_at_calvingfront_model_array]
        velocity_at_calvingfront_model_array_monthly =  velocity_at_calvingfront_model_array_flattened
        velocity_at_calvingfront_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in velocity_at_calvingfront_model_array_monthly]

        thickness_at_calvingfront_model_array_flattened = [sublist[0] for sublist in thickness_at_calvingfront_model_array]
        thickness_at_calvingfront_model_array_monthly =  thickness_at_calvingfront_model_array_flattened
        thickness_at_calvingfront_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in thickness_at_calvingfront_model_array_monthly]

        width_at_calvingfront_model_array_flattened = [sublist[0] for sublist in width_at_calvingfront_model_array]
        width_at_calvingfront_model_array_monthly =  width_at_calvingfront_model_array_flattened
        width_at_calvingfront_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in width_at_calvingfront_model_array_monthly]

        volume_bsl_model_array_flattened = [sublist[0] for sublist in volume_bsl_model_array]
        volume_bsl_model_array_monthly =  volume_bsl_model_array_flattened
        volume_bsl_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in volume_bsl_model_array_monthly]

        volume_bwl_model_array_flattened = [sublist[0] for sublist in volume_bwl_model_array]
        volume_bwl_model_array_monthly =  volume_bwl_model_array_flattened
        volume_bwl_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in volume_bwl_model_array_monthly]


        #pdb.set_trace()

        # calving_thickness_model_array_flattend = [sublist[0] for sublist in calving_thickness_model_array]
        # calving_thickness_model_array_monthly =  calving_thickness_model_array_flattend
        # calving_thickness_model_array_annual = [[np.mean(sublist[i:i + 12]) for i in range(0, len(sublist), 12)] for sublist in calving_thickness_model_array_monthly]
    else:
        lengthchange_m_TMS_model_array_annual = lengthchange_m_TMS_model_array
        lengthchange_dLdt_model_array_annual = lengthchange_dLdt_model_array
        calving_flux_Gta_TMS_model_array_annual = calving_flux_Gta_TMS_model_array
        massbalclim_TMS_model_array_annual = massbalclim_TMS_model_array
        massbaltotal_TMS_model_array_annual = massbaltotal_TMS_model_array
        massbalclim_TMS_model_array_annual_gta = massbalclim_Gta_TMS_model_array
        massbaltotal_TMS_model_array_annual_gta = massbaltotal_Gta_TMS_model_array
        FA_mwea_TMS_model_array_annual = FA_mwea_TMS_model_array
        area_km2_TMS_model_array_annual = area_km2_TMS_model_array
        velocity_at_calvingfront_model_array_annual = velocity_at_calvingfront_model_array
        thickness_at_calvingfront_model_array_annual = thickness_at_calvingfront_model_array
        width_at_calvingfront_model_array_annual = width_at_calvingfront_model_array
        volume_bsl_model_array_annual = volume_bsl_model_array
        volume_bwl_model_array_annual = volume_bwl_model_array


    #%% Run the particle batch smoother/AMIS
    
    # ===== Prepare the input for particle batch smoother as the m*1/m*N array ===== 

    # ===== Model =====
    # ----Convert and transpose lengthchange_m_TMS_model_array_annual if necessary
    lengthchange_m_TMS_model_array_annual = np.asarray(lengthchange_m_TMS_model_array_annual)
    if lengthchange_m_TMS_model_array_annual.ndim == 2:
        lengthchange_m_TMS_model_array_annual = lengthchange_m_TMS_model_array_annual.T

    # ----Convert and transpose lengthchange_dLdt_model_array_annual if necessary                
    # Ensure lengthchange_dLdt_model_array_annual is properly shaped before transposing
    lengthchange_dLdt_model_array_annual = np.asarray(lengthchange_dLdt_model_array_annual)
    if lengthchange_dLdt_model_array_annual.ndim == 2:
        lengthchange_dLdt_model_array_annual = lengthchange_dLdt_model_array_annual.T

    # ----Convert and transpose calving_flux_Gta_TMS_model_array_annual if necessary
    calving_flux_Gta_TMS_model_array_annual = np.asarray(calving_flux_Gta_TMS_model_array_annual)
    if calving_flux_Gta_TMS_model_array_annual.ndim == 2:
        calving_flux_Gta_TMS_model_array_annual = calving_flux_Gta_TMS_model_array_annual.T

    # #---- Convert and transpose massbalclim_TMS_model_array_annual if necessary
    massbalclim_TMS_model_array_annual = np.asarray(massbalclim_TMS_model_array_annual)
    if massbalclim_TMS_model_array_annual.ndim == 2:
        massbalclim_TMS_model_array_annual = massbalclim_TMS_model_array_annual.T
    
    # # ----Convert and transpose massbaltotal_TMS_model_array_annual if necessary
    massbaltotal_TMS_model_array_annual = np.asarray(massbaltotal_TMS_model_array_annual)
    if massbaltotal_TMS_model_array_annual.ndim == 2:
        massbaltotal_TMS_model_array_annual = massbaltotal_TMS_model_array_annual.T

    # # ----Convert and transpose massbalclim_TMS_model_array_annual_gta if necessary
    massbalclim_TMS_model_array_annual_gta = np.asarray(massbalclim_TMS_model_array_annual_gta)
    if massbalclim_TMS_model_array_annual_gta.ndim == 2:
        massbalclim_TMS_model_array_annual_gta = massbalclim_TMS_model_array_annual_gta.T

    # # ----Convert and transpose massbaltotal_TMS_model_array_annual_gta if necessary
    massbaltotal_TMS_model_array_annual_gta = np.asarray(massbaltotal_TMS_model_array_annual_gta)
    if massbaltotal_TMS_model_array_annual_gta.ndim == 2:
        massbaltotal_TMS_model_array_annual_gta = massbaltotal_TMS_model_array_annual_gta.T

    #---- Convert and transpose FA_mwea_TMS_model_array_annual if necessary
    FA_mwea_TMS_model_array_annual = np.asarray(FA_mwea_TMS_model_array_annual)
    if FA_mwea_TMS_model_array_annual.ndim == 2:
        FA_mwea_TMS_model_array_annual = FA_mwea_TMS_model_array_annual.T

    #---- Convert and transpose area_km2_timeseries_model_array_annual if necessary
    area_km2_TMS_model_array_annual = np.asarray(area_km2_TMS_model_array_annual)
    if area_km2_TMS_model_array_annual.ndim == 2:
        area_km2_TMS_model_array_annual = area_km2_TMS_model_array_annual.T    

    #---- Convert and transpose velocity_at_calvingfront_model_array_annual if necessary
    velocity_at_calvingfront_model_array_annual = np.asarray(velocity_at_calvingfront_model_array_annual)
    if velocity_at_calvingfront_model_array_annual.ndim == 2:
        velocity_at_calvingfront_model_array_annual = velocity_at_calvingfront_model_array_annual.T

    #---- Convert and transpose thickness_at_calvingfront_model_array_annual if necessary
    thickness_at_calvingfront_model_array_annual = np.asarray(thickness_at_calvingfront_model_array_annual)
    if thickness_at_calvingfront_model_array_annual.ndim == 2:
        thickness_at_calvingfront_model_array_annual = thickness_at_calvingfront_model_array_annual.T

    #---- Convert and transpose width_at_calvingfront_model_array_annual if necessary
    width_at_calvingfront_model_array_annual = np.asarray(width_at_calvingfront_model_array_annual)
    if width_at_calvingfront_model_array_annual.ndim == 2:
        width_at_calvingfront_model_array_annual = width_at_calvingfront_model_array_annual.T

    #---- Convert and transpose volume_bsl_model_array_annual if necessary
    volume_bsl_model_array_annual = np.asarray(volume_bsl_model_array_annual)
    if volume_bsl_model_array_annual.ndim == 2:
        volume_bsl_model_array_annual = volume_bsl_model_array_annual.T

    #---- Convert and transpose volume_bwl_model_array_annual if necessary
    volume_bwl_model_array_annual = np.asarray(volume_bwl_model_array_annual)
    if volume_bwl_model_array_annual.ndim == 2:
        volume_bwl_model_array_annual = volume_bwl_model_array_annual.T

    # ==== Replace the outliers by the boundarys
    #---- maskout the inf or -inf value based on the length change #TODO  Revise it , if it's inf, a specific number , e.g. 5000
    lengthchange_dLdt_model_array_annual = np.clip(lengthchange_dLdt_model_array_annual, min_length_change_myr, max_length_change_myr)


    
    # === store the model output
    if store_result:

        # store the monthly results
        if store_monthly_step:

            # remove outliers based on the length change rate 
            lengthchange_dLdt_model_array_monthly = np.array(lengthchange_dLdt_model_array_monthly)
            lengthchange_m_TMS_model_array_monthly = np.array(lengthchange_m_TMS_model_array_monthly)
            calving_flux_Gta_TMS_model_array_monthly = np.array(calving_flux_Gta_TMS_model_array_monthly)
            FA_mwea_TMS_model_array_monthly = np.array(FA_mwea_TMS_model_array_monthly)
            area_km2_TMS_model_array_monthly = np.array(area_km2_TMS_model_array_monthly)
            velocity_at_calvingfront_model_array_monthly = np.array(velocity_at_calvingfront_model_array_monthly)
            thickness_at_calvingfront_model_array_monthly = np.array(thickness_at_calvingfront_model_array_monthly)
            width_at_calvingfront_model_array_monthly = np.array(width_at_calvingfront_model_array_monthly)
            volume_bsl_model_array_monthly = np.array(volume_bsl_model_array_monthly)
            volume_bwl_model_array_monthly = np.array(volume_bwl_model_array_monthly)
            massbalclim_TMS_model_array_monthly = np.array(massbalclim_TMS_model_array_monthly)
            massbaltotal_TMS_model_array_monthly =np.array( massbaltotal_TMS_model_array_monthly)
            massbalclim_TMS_model_array_monthly_gta = np.array(massbalclim_TMS_model_array_monthly_gta)
            massbaltotal_TMS_model_array_monthly_gta = np.array(massbaltotal_TMS_model_array_monthly_gta)

            # Dictionary to store dataset names and corresponding data arrays
            output_data_dict_monthly = {
                'calving_flux_Gta_average_model_array': calving_flux_Gta_average_model_array,
                'massbalclim_model_array': massbalclim_model_array,
                'massbaltotal_model_array': massbaltotal_model_array,
                'massbalclim_Gta_model_array': massbalclim_Gta_model_array,
                'massbaltotal_Gta_model_array': massbaltotal_Gta_model_array,
                'FA_mwea_average_model_array': FA_mwea_average_model_array,
                'calving_thickness_model_array': calving_thickness_model_array,
                'lengthchange_dLdt_model_array_monthly': lengthchange_dLdt_model_array_monthly,
                'lengthchange_m_TMS_model_array_monthly': lengthchange_m_TMS_model_array_monthly,
                'calving_flux_Gta_TMS_model_array_monthly': calving_flux_Gta_TMS_model_array_monthly,
                'FA_mwea_TMS_model_array_monthly': FA_mwea_TMS_model_array,
                'area_km2_TMS_model_array_monthly': area_km2_TMS_model_array_monthly,
                'velocity_at_calvingfront_model_array_monthly': velocity_at_calvingfront_model_array_monthly,
                'thickness_at_calvingfront_model_array_monthly': thickness_at_calvingfront_model_array_monthly,
                'width_at_calvingfront_model_array_monthly': width_at_calvingfront_model_array_monthly,
                'volume_bsl_model_array_monthly': volume_bsl_model_array_monthly,
                'volume_bwl_model_array_monthly': volume_bwl_model_array_monthly,
                'massbalclim_TMS_model_array_monthly': massbalclim_TMS_model_array_monthly,
                'massbaltotal_TMS_model_array_monthly': massbaltotal_TMS_model_array_monthly,
                'massbalclim_TMS_model_array_monthly_gta': massbalclim_TMS_model_array_monthly_gta,
                'massbaltotal_TMS_model_array_monthly_gta': massbaltotal_TMS_model_array_monthly_gta
            }

            # store the monthly results in json
            output_folder_monthly = os.path.join(save_path_modeloutput_glac,'Monthly')  # Assuming `pygem_prms.output_fp` exists
            os.makedirs(output_folder_monthly, exist_ok=True)  # Assuming `os.makedirs` exists
            output_filename_monthly = f"calibration_model_Monthly_output_{rgiid_ind}_{N_iteration}.json"  # Assuming `pygem_prms.output_fp` exists
            output_fp_monthly = os.path.join(output_folder_monthly, output_filename_monthly)

            # Save to json
            with open(output_fp_monthly, 'w') as f:
                json.dump(output_data_dict_monthly, f, indent=4, default=convert_to_serializable)

        # store the annual results
        # Dictionary to store dataset names and corresponding data arrays
        output_data_dict_annual = {
                                'lengthchange_dLdt_model_array_annual_myr': lengthchange_dLdt_model_array_annual,
                                'lengthchange_m_TMS_model_array_annual': lengthchange_m_TMS_model_array_annual,
                                'calving_flux_Gta_TMS_model_array_annual': calving_flux_Gta_TMS_model_array_annual,
                                'massbalclim_TMS_model_array_annual_mwea': massbalclim_TMS_model_array_annual,
                                'massbaltotal_TMS_model_array_annual_mwea': massbaltotal_TMS_model_array_annual,
                                'massbalclim_TMS_model_array_annual_gta': massbalclim_TMS_model_array_annual_gta,
                                'massbaltotal_TMS_model_array_annual_gta': massbaltotal_TMS_model_array_annual_gta,
                                'FA_mwea_TMS_model_array_annual': FA_mwea_TMS_model_array_annual,
                                'area_km2_timeseries_model_array_annual': area_km2_TMS_model_array_annual,
                                'velocity_at_calvingfront_model_array_annual_myr': velocity_at_calvingfront_model_array_annual,
                                'thickness_at_calvingfront_model_array_annual_m': thickness_at_calvingfront_model_array_annual,
                                'width_at_calvingfront_model_array_annual_m': width_at_calvingfront_model_array_annual,
                                'volume_bsl_model_array_annual_m3': volume_bsl_model_array_annual,
                                'volume_bwl_model_array_annual_m3': volume_bwl_model_array_annual,
                                'calving_flux_Gta_average_model_array': calving_flux_Gta_average_model_array,
                                'calving_thickness_model_array_m': calving_thickness_model_array,
                                'massbalclim_model_array_mwea': massbalclim_model_array,
                                'massbaltotal_model_array_mwea': massbaltotal_model_array,
                                'massbalclim_model_array_Gta': massbalclim_Gta_model_array,
                                'massbaltotal_model_array_Gta': massbaltotal_Gta_model_array,
                                'FA_mwea_average_model_array': FA_mwea_average_model_array
                                }
        # save the weighted information in a hdf5 file
        output_folder_annual = os.path.join(save_path_modeloutput_glac, 'Annual')  # Assuming `pygem_prms.output_fp` exists
        os.makedirs(output_folder_annual, exist_ok=True)  # Assuming `os.makedirs` exists
        output_filename_annual = f"calibration_model_Annual_output_{rgiid_ind}_{N_iteration}.json" # dataset with weights and removed outliers compared to the prior samples/values
        output_fp_annual = os.path.join(output_folder_annual, output_filename_annual)
        # Save to JSON
        with open(output_fp_annual, 'w') as f:
            json.dump(output_data_dict_annual, f, indent=4, default=convert_to_serializable)

    # return
    if return_all:
    # here is the full returen of the model, #TODO add the monthly results, or depends on the user's choice
        return (
            lengthchange_dLdt_model_array_annual,lengthchange_m_TMS_model_array_annual,calving_flux_Gta_TMS_model_array_annual,
            massbalclim_TMS_model_array_annual,massbaltotal_TMS_model_array_annual,
            massbalclim_TMS_model_array_annual_gta,massbaltotal_TMS_model_array_annual_gta,FA_mwea_TMS_model_array_annual,
            area_km2_TMS_model_array_annual,velocity_at_calvingfront_model_array_annual,thickness_at_calvingfront_model_array_annual,
            width_at_calvingfront_model_array_annual,volume_bsl_model_array_annual,volume_bwl_model_array_annual,
            calving_flux_Gta_average_model_array,calving_thickness_model_array, massbalclim_model_array,massbaltotal_model_array,
            massbalclim_Gta_model_array,massbaltotal_Gta_model_array,
            FA_mwea_average_model_array,mb_obs_mwea,mb_obs_mwea_err
        )
    else:
        return lengthchange_dLdt_model_array_annual, calving_flux_Gta_TMS_model_array_annual,calving_flux_Gta_average_model_array,massbalclim_TMS_model_array_annual,massbalclim_model_array,mb_obs_mwea,mb_obs_mwea_err
    


def cali_PBS_MB_FA_RT(regions, args, frontalablation_fp='', frontalablation_fn='',
                        frontalablation_annual_fp = '', frontalablation_annual_fn = '',
                        output_fp='', hugonnet_fp='',hugonnet_fn='',lengthchange_annual_fp='',
                        lengthchange_annual_fn='',verbose=False,overwrite = False,Visualize_Index = True,debug = True,
                        store_monthly_step = True,log_level = logging.INFO):
    """
    Calibration of the mass balance and frontal ablation model
    Parameters
    ----------
    regions : list
        list of regions to be calibrated
    args : argparse.Namespace
        arguments provided by the command line
    frontalablation_fp : str, optional
        file path to the frontal ablation data, by default ''
    frontalablation_fn : str, optional
        file name of the frontal ablation data, by default ''
    output_fp : str, optional
        output file path, by default ''
    hugonnet2021_fp : str, optional
        file path to the Hugonnet et al. 2021 data, by default ''
    hugonnet2021_fn : str, optional
        file name of the Hugonnet et al. 2021 data, by default ''
    lengthchange_annual_fp : str, optional
        file path to the annual length change data, by default ''
    lengthchange_annual_fn : str, optional
        file name of the annual length change data, by default ''
    verbose : bool, optional
        whether to print out the information, by default False
    overwrite : bool, optional
        whether to overwrite the existing files, by default False
    Visualize_Index : bool, optional
        whether to visualize the particles and weighted results, by default False
    store_monthly_step : bool, optional
        whether to store the monthly step data, by default True
    log_level : str, optional
        log level, by default 'INFO', can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    Returns
    -------
    None
    """
    # load the rgi glacier id if it's provided
    if args.rgi_glac_number != '':
        rgi_glac_number_single = args.rgi_glac_number
    else:
        rgi_glac_number_single = None
    # ===== Load mass balance and frontal ablation data, and length change data =====
    #Load calving glacier data (20 years averaged data,i.e. one value for the 20-year period) ===== 
    fa_glac_data = pd.read_csv(frontalablation_fp + frontalablation_fn)
    # Load the annual frontal ablation data #TODO At the moment, we don't have the annual frontal ablation data, so we use the 20-year averaged data
    csv_path_FA_annual = frontalablation_annual_fp + frontalablation_annual_fn
    if os.path.exists(csv_path_FA_annual):
        frontalablation_annual_data = pd.read_csv(csv_path_FA_annual)
        frontalablation_annual_data['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in frontalablation_annual_data.RGIId.values]
    else:
        frontalablation_annual_data = None
        print(f"File not found: {csv_path_FA_annual}. Skipping this step.")
    #TODO Check the dataset, which should be the climatic mass balance, not the total mass balance
    mb_data = pd.read_csv(hugonnet_fp + hugonnet_fn)
    fa_glac_data['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in fa_glac_data.RGIId.values]
    #Load lenght change data ===== 
    #TODO Set the condition for calibtation variables, calibrate FA or dLdt or both, and the condition for monthly or annual calibration
    lengthchange_annual_data = pd.read_csv(lengthchange_annual_fp + lengthchange_annual_fn)
    lengthchange_annual_data['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in lengthchange_annual_data.RGIId.values]

    #%% ===== Regional calibration =====
    for reg in [regions]:
        # skip over any regions we don't have data for
        if reg not in fa_glac_data['O1Region'].values.tolist():
            continue
        output_fn = str(reg) + '-calving_cal_ind.csv'

        # set the output file path
        save_path_figure_reg = os.path.join(save_path_figure, f"{str(reg).zfill(2)}/")
        os.makedirs(save_path_figure_reg, exist_ok=True)  # Safe for concurrent runs
        save_path_parameter_reg = os.path.join(save_path_parameter, f"{str(reg).zfill(2)}/")
        os.makedirs(save_path_parameter_reg, exist_ok=True)  # Safe for concurrent runs
        save_path_modeloutput_reg = os.path.join(save_path_modeloutput, f"{str(reg).zfill(2)}/")
        os.makedirs(save_path_modeloutput_reg, exist_ok=True)  # Safe for concurrent runs
        save_path_AMISINFO_reg = os.path.join(save_path_AMISINFO, f"{str(reg).zfill(2)}/")
        os.makedirs(save_path_AMISINFO_reg, exist_ok=True)  # Safe for concurrent runs
        save_path_log_reg = os.path.join(save_path_log, f"{str(reg).zfill(2)}/")
        os.makedirs(save_path_log_reg, exist_ok=True)  # Safe for concurrent runs
        save_path_statistics_reg = os.path.join(save_path_statistics, f"{str(reg).zfill(2)}/")
        os.makedirs(save_path_statistics_reg, exist_ok=True)  # Safe for concurrent runs
        floating_info_fp_reg = os.path.join(floating_info_fp, f"{str(reg).zfill(2)}/")
        os.makedirs(floating_info_fp_reg, exist_ok=True)  # Safe for concurrent runs

        # === Regional data ===
        fa_glac_data_reg = fa_glac_data.loc[fa_glac_data['O1Region'] == reg, :].copy()
        fa_glac_data_reg.reset_index(inplace=True, drop=True)

        lengthchange_annual_data_reg = lengthchange_annual_data.loc[lengthchange_annual_data['O1Region'] == reg, :].copy()
        lengthchange_annual_data_reg.reset_index(inplace=True, drop=True)

        fa_glac_data_reg['glacno'] = ''

        for nglac, rgiid in enumerate(fa_glac_data_reg.RGIId):
            # Avoid regional data and observations from multiple RGIIds (len==14)
            if not fa_glac_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_glac_data_reg.loc[nglac,'RGIId']) == 14:
                fa_glac_data_reg.loc[nglac,'glacno'] = (str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
                                                        rgiid.split('-')[1].split('.')[1])
                
        for nglac, rgiid in enumerate(lengthchange_annual_data_reg.RGIId):
            # Avoid regional data and observations from multiple RGIIds (len==14)
            if not lengthchange_annual_data_reg.loc[nglac,'RGIId'] == 'all' and len(lengthchange_annual_data_reg.loc[nglac,'RGIId']) == 14:
                lengthchange_annual_data_reg.loc[nglac,'glacno'] = (str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
                                                        rgiid.split('-')[1].split('.')[1])
        if frontalablation_annual_data is not None:
            fa_annual_data_reg = frontalablation_annual_data.loc[frontalablation_annual_data['O1Region'] == reg, :].copy()
            fa_annual_data_reg.reset_index(inplace=True, drop=True)       
            for nglac, rgiid in enumerate(fa_annual_data_reg.RGIId):
                # Avoid regional data and observations from multiple RGIIds (len==14)
                if not fa_annual_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_annual_data_reg.loc[nglac,'RGIId']) == 14:
                    fa_annual_data_reg.loc[nglac,'glacno'] = (str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
                                                            rgiid.split('-')[1].split('.')[1])
            fa_annual_data_reg = fa_annual_data_reg.dropna(axis=0, subset=['glacno'])
            fa_annual_data_reg.reset_index(inplace=True, drop=True)
            glacno_reg_wdata_FA_annual = sorted(list(fa_annual_data_reg.glacno.values)) # annuall timeseries data
        else:
            fa_annual_data_reg = None
            glacno_reg_wdata_FA_annual = None
        # ===== Drop observations that aren't of individual glaciers, remove thoese with nan glacno
        fa_glac_data_reg = fa_glac_data_reg.dropna(axis=0, subset=['glacno'])
        fa_glac_data_reg.reset_index(inplace=True, drop=True)
        lengthchange_annual_data_reg = lengthchange_annual_data_reg.dropna(axis=0, subset=['glacno'])
        lengthchange_annual_data_reg.reset_index(inplace=True, drop=True)
        if verbose:
            print('fa_glac_data_reg 1st:',fa_glac_data_reg)
            print('lengthchange_annual_data_reg 1st:',lengthchange_annual_data_reg)
            print('fa_annual_data_reg 1st:',fa_annual_data_reg)

        # ===== regional observations
        reg_calving_gta_obs = fa_glac_data_reg['fa_gta_obs'].sum()
        # Glacier numbers for model runs
        if rgi_glac_number_single is not None:
            glacno_reg_wdata = [rgi_glac_number_single]
        else:
            #TODO Set the condition for calibtation variables, calibrate FA or dLdt or both;Maybe add the regional mass balance total, climatic mass balance here we assume all glaciers has the MB data
            glacno_reg_wdata_FA = sorted(list(fa_glac_data_reg.glacno.values)) # 20 years averaged data
            glacno_reg_wdata_dLdt_annual = sorted(list(lengthchange_annual_data_reg.glacno.values)) # annuall timeseries data       
            glacno_reg_wdata = sorted(list(set(glacno_reg_wdata_FA).intersection(set(glacno_reg_wdata_dLdt_annual))))
            # glacno_reg_wdata = sorted(list(set(glacno_reg_wdata_FA_annual).intersection(set(glacno_reg_wdata_dLdt_annual))))
        print('glacno_reg_wdata:', glacno_reg_wdata)
        print('type of glacno_reg_wdata:', type(glacno_reg_wdata))
        # ===== LOAD GLACIERS =====
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no = glacno_reg_wdata) # TODO check the input of function selectglaciersrgitable
        # Select Tidewater glaciers
        termtype_list = [1,5] # TODO check the termtype list, at the moment, either RGI6 or RGI7, has the correct and assigned termtype, we based on the Williams et al. 2020 for the Northern Hemisphere
        # main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi_all['TermType'] = 1
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi.reset_index(inplace=True, drop=True)

        # ----- QUALITY CONTROL USING MB_CLIM COMPARED TO REGIONAL MASS BALANCE -----
        mb_data['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in mb_data.RGIId.values]
        mb_data_reg = mb_data.loc[mb_data['O1Region'] == reg, :]
        mb_data_reg.reset_index(inplace=True)

        mb_clim_reg_avg = np.mean(mb_data_reg.mb_mwea)
        mb_clim_reg_std = np.std(mb_data_reg.mb_mwea)
        mb_clim_reg_3std_max = mb_clim_reg_avg + 3*mb_clim_reg_std
        mb_clim_reg_max = np.max(mb_data_reg.mb_mwea)
        mb_clim_reg_3std_min = mb_clim_reg_avg - 3*mb_clim_reg_std
        if verbose:
            print('mb_clim_reg_avg:', np.round(mb_clim_reg_avg,2), '+/-', np.round(mb_clim_reg_std,2))
            print('mb_clim_3std (neg):', np.round(mb_clim_reg_3std_min,2))
            print('mb_clim_3std (pos):', np.round(mb_clim_reg_3std_max,2))
            print('mb_clim_min:', np.round(mb_data_reg.mb_mwea.min(),2))
            print('mb_clim_max:', np.round(mb_clim_reg_max,2))

        # ===== Calibrate individuals =====
        
        #%% Prepare the output dataframe
        output_cns = ['RGIId', 'calving_k', 'calving_k_nmad', 'calving_thick', 'calving_flux_Gta', 'fa_gta_obs', 'fa_gta_obs_unc', 'fa_gta_max', 
                        'no_errors', 'oggm_dynamics', 
                        'mb_clim_gta', 'mb_total_gta', 'mb_clim_mwea', 'mb_total_mwea','length_change_ma_obs',
                        'length_change_ma_obs_unc']
        
        output_df_all = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cns))), columns=output_cns)
        output_df_all['RGIId'] = main_glac_rgi.RGIId
        output_df_all['calving_k_nmad'] = 0

        #%%
        # Load observations 
        fa_obs_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg['fa_gta_obs']))
        fa_obs_unc_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg['fa_gta_obs_unc']))
        lengthchange_obs_dict = dict(zip(lengthchange_annual_data_reg.RGIId, lengthchange_annual_data_reg['dLdt_m_per_yr']))
        lengthchange_obs_unc_dict = dict(zip(lengthchange_annual_data_reg.RGIId, lengthchange_annual_data_reg['dLdt_m_per_yr_unc']))
        # Set the output of the model, about the observations 
        output_df_all['fa_gta_obs'] = output_df_all['RGIId'].map(fa_obs_dict)
        output_df_all['fa_gta_obs_unc'] = output_df_all['RGIId'].map(fa_obs_unc_dict)
        output_df_all['length_change_ma_obs'] = output_df_all['RGIId'].map(lengthchange_obs_dict)
        output_df_all['length_change_ma_obs_unc'] = output_df_all['RGIId'].map(lengthchange_obs_unc_dict)

        if frontalablation_annual_data is not None:
            fa_obs_annual_dict = dict(zip(fa_annual_data_reg.RGIId, fa_annual_data_reg['fa_Gta_annual']))
            fa_obs_unc_annual_dict = dict(zip(fa_annual_data_reg.RGIId, fa_annual_data_reg['fa_Gta_unc_annual']))
            output_df_all['fa_gta_obs_annual'] = output_df_all['RGIId'].map(fa_obs_annual_dict)
            output_df_all['fa_gta_obs_unc_annual'] = output_df_all['RGIId'].map(fa_obs_unc_annual_dict)
        
        #fa_glacname_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.glacier_name))
        #output_df_all['name'] = output_df_all['RGIId'].map(fa_glacname_dict)
        rgi_area_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Area))
        output_df_all['area_km2'] = output_df_all['RGIId'].map(rgi_area_dict)
        # TODO add the observation of mass balance (Climatic mass balance)



        # ----- LOAD DATA ON MB_CLIM CORRECTED FOR FRONTAL ABLATION -----
        # use this to assess reasonableness of results and see if calving_k values affected
        fa_rgiids_list = list(fa_glac_data_reg.RGIId)
        output_df_all['mb_total_gta_obs'] = np.nan
        output_df_all['mb_clim_gta_obs'] = np.nan
        output_df_all['mb_total_mwea_obs'] = np.nan
        output_df_all['mb_clim_mwea_obs'] = np.nan
        #output_df_all['thick_measured_yn'] = np.nan
        for nglac, rgiid in enumerate(list(output_df_all.RGIId)):
            fa_idx = fa_rgiids_list.index(rgiid)
            output_df_all.loc[nglac, 'mb_total_gta_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_gta_mbtot']
            output_df_all.loc[nglac, 'mb_clim_gta_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_gta_mbclim']
            output_df_all.loc[nglac, 'mb_total_mwea_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_mwea_mbtot']
            output_df_all.loc[nglac, 'mb_clim_mwea_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_mwea_mbclim']
        # output_df_all.loc[nglac, 'thick_measured_yn'] = fa_glac_data_reg.loc[fa_idx, 'thick_measured_yn']
        # ----- CORRECT TOO POSITIVE CLIMATIC MASS BALANCES -----
        output_df_all['mb_clim_gta'] = output_df_all['mb_clim_gta_obs']
        output_df_all['mb_total_gta'] = output_df_all['mb_total_gta_obs']
        output_df_all['mb_clim_mwea'] = output_df_all['mb_clim_mwea_obs']
        output_df_all['mb_total_mwea'] = output_df_all['mb_total_mwea_obs']
        output_df_all['fa_gta_max'] = output_df_all['fa_gta_obs']
        
        output_df_badmbclim = output_df_all.loc[output_df_all.mb_clim_mwea_obs > mb_clim_reg_3std_max]
        # Correct by using mean + 3std as maximum climatic mass balance
        if output_df_badmbclim.shape[0] > 0:
            #print("*************there are bad climate balance, which is lager than the region mean+3std")
            rgiids_toopos = list(output_df_badmbclim.RGIId)

            for nglac, rgiid in enumerate(list(output_df_all.RGIId)):
                if rgiid in rgiids_toopos:
                    # Specify maximum frontal ablation based on maximum climatic mass balance
                    mb_clim_mwea = mb_clim_reg_3std_max
                    area_m2 = output_df_all.loc[nglac,'area_km2'] * 1e6
                    mb_clim_gta = mwea_to_gta(mb_clim_mwea, area_m2)

                    mb_total_gta = output_df_all.loc[nglac,'mb_total_gta_obs']
            
                    fa_gta_max = mb_clim_gta - mb_total_gta
                
                    output_df_all.loc[nglac,'fa_gta_max'] = fa_gta_max
                    output_df_all.loc[nglac,'mb_clim_mwea'] = mb_clim_mwea
                    output_df_all.loc[nglac,'mb_clim_gta'] = mb_clim_gta

        # ----- classes the glaciers, for  failed, good AMIS implantation, bad but arrived the maximum iterations, and account the numbers -----
        Failed_glacs = []
        Good_AMIS = []
        Bad_AMIS =[] # bad AMIS, but arrived the maximum iterations
        N_failed =0
        N_good_AMIS = 0
        N_bad_AMIS = 0
        N_good_iterations = []
        N_bad_iterations = []

        # ----- RUN THE CALIBRATION -----
        #TODO It weights Parameters BASED ON INDIVIDUAL GLACIER MASS BALANCE + dLdt + FRONTAL ABLATION DATA, but at the moment,it's Monte Carlo -----
        for nglac in np.arange(main_glac_rgi.shape[0]):
            try:
                glacier_str = '{0:0.5f}'.format(main_glac_rgi.loc[nglac,'RGIId_float'])
                #if main_glac_rgi.loc[nglac,'RGIId'] in ['RGI60-03.00108']:

                # Construct the glacier-specific directory path
                save_path_figure_glac = os.path.join(save_path_figure_reg, glacier_str)
                os.makedirs(save_path_figure_glac, exist_ok=True)
                save_path_parameter_glac = os.path.join(save_path_parameter_reg, glacier_str)
                os.makedirs(save_path_parameter_glac, exist_ok=True)
                save_path_modeloutput_glac = os.path.join(save_path_modeloutput_reg, glacier_str)
                os.makedirs(save_path_modeloutput_glac, exist_ok=True)
                save_path_AMISINFO_glac = os.path.join(save_path_AMISINFO_reg, glacier_str)
                os.makedirs(save_path_AMISINFO_glac, exist_ok=True)
                save_path_log_glac = os.path.join(save_path_log_reg, glacier_str)
                os.makedirs(save_path_log_glac, exist_ok=True)
                floating_info_fp_glac = os.path.join(floating_info_fp_reg, glacier_str)
                os.makedirs(floating_info_fp_glac, exist_ok=True)
                # Select individual glacier
                main_glac_rgi_ind = main_glac_rgi.loc[[nglac],:]
                main_glac_rgi_ind.reset_index(inplace=True, drop=True)
                rgiid_ind = main_glac_rgi_ind.loc[0,'RGIId']

                fa_glac_data_ind = fa_glac_data_reg.loc[fa_glac_data_reg.RGIId == rgiid_ind, :]
                fa_glac_data_ind.reset_index(inplace=True, drop=True)
                fa_gta_obs_ind = fa_glac_data_ind.loc[0,'fa_gta_obs']
                fa_gta_obs_unc_ind = fa_glac_data_ind.loc[0,'fa_gta_obs_unc']
                lengthchange_annual_data_ind = lengthchange_annual_data_reg.loc[lengthchange_annual_data_reg.RGIId == rgiid_ind, :]
                lengthchange_annual_data_ind.reset_index(inplace=True,drop=True)
                lengthchange_dLdt_obs_ind = ast.literal_eval(lengthchange_annual_data_ind.loc[0,'dLdt_m_per_yr'])
                lengthchnage_dLdt_unc_obs_ind = ast.literal_eval(lengthchange_annual_data_ind.loc[0,'dLdt_m_per_yr_unc'])
                if frontalablation_annual_data is not None:
                    fa_annual_data_ind = fa_annual_data_reg.loc[fa_annual_data_reg.RGIId == rgiid_ind, :]
                    fa_annual_data_ind.reset_index(inplace=True, drop=True)
                    fa_annual_data_obs_ind = ast.literal_eval(fa_annual_data_ind.loc[0,'fa_Gta_annual'])
                    fa_annual_data_obs_unc_ind = ast.literal_eval(fa_annual_data_ind.loc[0,'fa_Gta_unc_annual'])
                else:
                    fa_annual_data_obs_ind = np.nan
                    fa_annual_data_obs_unc_ind = np.nan
                #TODO add the mass balance information for ind

                # Quantify the fa by the fa_gta_max, if the fa_gta_obs is larger than the fa_gta_max, then set the fa_gta_obs as fa_gta_max
                fa_gta_max = output_df_all.loc[nglac,'fa_gta_max']
                fa_gta_obs_unc = output_df_all.loc[nglac,'fa_gta_obs_unc']
                print('The glacier is:',rgiid_ind,'the max FA gta is:',fa_gta_max)
                if fa_glac_data_ind.loc[0,'fa_gta_obs'] > fa_gta_max:
                    reg_calving_gta_obs = fa_gta_max
                    fa_glac_data_ind.loc[0,'fa_gta_obs'] = fa_gta_max


                
                # ===== generate the particles =====
                max_iterations = pygem_prms.max_iterations
                if (max_iterations < 1 ):
                    print("INVALID NUMBER OF MAX_ITERATIONS {0} < 1".format(max_iterations))
                    raise ValueError("INVALID NUMBER OF MAX_ITERATIONS")
                Sample_N = pygem_prms.pbs_sample_no
                parameters_keys = pygem_prms.vars_to_calibrate
                for j in range(max_iterations): #TODO DO THE APATED CALIBRATION HERE
                    #Sample the parameters
                    print("the iteration is ",j)
                    #pdb.set_trace()    
                    # Generate the parameters
                    if j == 0:
                        # generate the parameters from the prior distribution
                        
                        parameters_dict = sample_prior(Sample_N)
                    else:
                        #Sample_N = Ne
                        #pdb.set_trace()
                        # generate the parameters from the posterior distribution (the last step)
                        parameters_values = thetap
                        parameters_dict = {key: value for key, value in zip(parameters_keys, parameters_values)}
                        parameters_dict['index'] = np.arange(Sample_N)
                            

                    # Run the coupled model (PyGEM_OGGM_SERMeQ)
                    lengthchange_dLdt_model_array_annual, calving_flux_Gta_TMS_model_array_annual,calving_flux_Gta_average_model_array,massbalclim_TMS_model_array_annual,massbalclim_model_array,mb_obs_mwea,mb_obs_mwea_err = Model_MB_FA_RT(model_function = reg_calving_flux,parameters_dict= parameters_dict,
                                                                                        rgiid_ind = rgiid_ind,main_glac_rgi = main_glac_rgi_ind,fa_glac_data_reg= fa_glac_data_ind,
                                                                                        ignore_nan=False,calibrate_timeseries =True,store_monthly_step=store_monthly_step,
                                                                                        return_all = False,store_result= True,N_iteration = j,save_path_figure_glac = save_path_figure_glac,
                                                                                        save_path_parameter_glac = save_path_parameter_glac,save_path_modeloutput_glac = save_path_modeloutput_glac,
                                                                                        save_path_log_glac = save_path_log_glac,log_level = log_level,floating_info_fp_glac = floating_info_fp_glac)

                    # ==== Replace the outliers by the boundarys
                    #---- maskout the inf or -inf value based on the length change #TODO  Revise it , if it's inf, a specific number , e.g. 5000
                    lengthchange_dLdt_model_array_annual = np.clip(lengthchange_dLdt_model_array_annual, min_length_change_myr, max_length_change_myr)
                    # ==== generate the array as the input for AMIS
                    #pdb.set_trace() # for 20-year 
                    lengthchange_dLdt_MB_model_array_annual_array = np.concatenate((lengthchange_dLdt_model_array_annual,massbalclim_model_array.T),axis = 0)                    
                    lengthchange_dLdt_model_array_annual_00_10 = lengthchange_dLdt_model_array_annual[:10,:]
                    lengthchange_dLdt_model_array_annual_11_20 = lengthchange_dLdt_model_array_annual[10:20,:]
                    massbalclim_TMS_model_array_annual_00_10 = massbalclim_TMS_model_array_annual[:10,:]
                    massbalclim_TMS_model_array_annual_11_20 = massbalclim_TMS_model_array_annual[10:20,:]
                    massbalclim_model_array_00_10 = np.expand_dims(np.nanmean(massbalclim_TMS_model_array_annual_00_10, axis=0),axis=1)
                    massbalclim_model_array_11_20 = np.expand_dims(np.nanmean(massbalclim_TMS_model_array_annual_11_20, axis=0),axis =1)
                    lengthchange_dLdt_MB_model_array_annual_array_00_10 = np.concatenate((lengthchange_dLdt_model_array_annual_00_10,massbalclim_model_array_00_10.T),axis = 0)
                    lengthchange_dLdt_MB_model_array_annual_array_11_20 = np.concatenate((lengthchange_dLdt_model_array_annual_11_20,massbalclim_model_array_11_20.T),axis = 0)
                    

                    # ===== Observations =====
                    # ----onvert to NumPy array and ensure proper shape of the observations #TODO At the moment, the calibration is based on the length change (TMS), the mass balance (multiple year average), and more choice can be added in the future
                    if j ==0:

                        ## ==== read the observation ====
                        lengthchange_dLdt_obs_ind_array = np.array(lengthchange_dLdt_obs_ind).reshape(len(lengthchange_dLdt_obs_ind),1)
                        mb_obs_mwea_ind_array = np.asarray(mb_obs_mwea).reshape(-1, 1)
                        fa_gta_obs_ind_array = np.asarray(fa_gta_obs_ind).reshape(-1, 1)
                        
                        # ----square of uncertainty of observations
                        lengthchnage_dLdt_unc_obs_ind_2_array = np.square(np.asarray(lengthchnage_dLdt_unc_obs_ind)).reshape(-1, 1)
                        mb_obs_mwea_err_2_array = np.square(np.asarray(mb_obs_mwea_err)).reshape(-1, 1)
                        fa_gta_obs_unc_2_array = np.square(np.asarray(fa_gta_obs_unc)).reshape(-1, 1)
                        if frontalablation_annual_data is not None:
                            fa_annual_data_obs_ind_array = np.array(fa_annual_data_obs_ind).reshape(len(fa_annual_data_obs_ind),1)
                            fa_annual_data_obs_unc_2_array = np.square(np.asarray(fa_annual_data_obs_unc_ind)).reshape(-1, 1)
                        #TODO At the momment, the calibration based on the length change, the mass balance, and the frontal ablation is not considered, and 
                        # for the length change, the first 10 years + the 10-year averages as the calibration data, and the last 10 years as the validation data,
                        # but for the future, it should has more flexiale choices, and the timeseries of mass balance and the frontal ablation should be considered as well.
                        # 10-years annual length change and the 10-year average climatic mass balance
                        lengthchange_dLdt_annual_CMB_obs_ind_array = np.concatenate((lengthchange_dLdt_obs_ind_array,mb_obs_mwea_ind_array),axis = 0)
                        lengthchange_dLdt_annual_CMB_obs_ind_2_array = np.concatenate((lengthchnage_dLdt_unc_obs_ind_2_array,mb_obs_mwea_err_2_array),axis = 0)

                        ##  ==== split the observation data as calibration data and validation data
                        #TODO At the moment, the calibration is based on the length change (TMS), the mass balance (multiple year average), and more choice can be added in the future
                        # for the length change, the first 10 years + the 10-year averages as the calibration data, and the last 10 years as the validation data,
                        # but for the future, it should has more flexiale choices, and the timeseries of mass balance and the frontal ablation should be considered as well.
                        # 10-years annual length change and the 10-year average climatic mass balance
                        #pdb.set_trace()
                        lengthchange_dLdt_obs_ind_array_00_10 = lengthchange_dLdt_obs_ind_array[:10, :]
                        lengthchange_dLdt_obs_ind_array_11_20 = lengthchange_dLdt_obs_ind_array[10:20,:]
                        lengthchange_dLdt_annual_CMB_obs_ind_array_00_10 = np.concatenate((lengthchange_dLdt_obs_ind_array_00_10,mb_obs_mwea_ind_array),axis = 0)
                        lengthchange_dLdt_annual_CMB_obs_ind_array_11_20 = np.concatenate((lengthchange_dLdt_obs_ind_array_11_20,mb_obs_mwea_ind_array),axis = 0)
                        lengthchnage_dLdt_unc_obs_ind_2_array_00_10 = lengthchnage_dLdt_unc_obs_ind_2_array[:10, :]
                        lengthchnage_dLdt_unc_obs_ind_2_array_11_20 = lengthchnage_dLdt_unc_obs_ind_2_array[10:20,:]
                        lengthchange_dLdt_annual_CMB_obs_ind_2_array_00_10 = np.concatenate((lengthchnage_dLdt_unc_obs_ind_2_array_00_10,mb_obs_mwea_err_2_array),axis = 0)
                        lengthchange_dLdt_annual_CMB_obs_ind_2_array_11_20 = np.concatenate((lengthchnage_dLdt_unc_obs_ind_2_array_11_20,mb_obs_mwea_err_2_array),axis = 0)


                    #%% Generate the proposal, and do the AMIS,
                    #TODO the validation shold be adjusted based on the user's choice, flexiblely based on the observation data, here is hard coded as the 10/20 years
                    # calibration dataset 2000-2009
                    predicted = lengthchange_dLdt_MB_model_array_annual_array_00_10
                    observations_sbst_masked = lengthchange_dLdt_annual_CMB_obs_ind_array_00_10
                    r_cov = lengthchange_dLdt_annual_CMB_obs_ind_2_array_00_10
                    # calibration dataset 2010-2019
                    # predicted = lengthchange_dLdt_MB_model_array_annual_array_11_20
                    # observations_sbst_masked = lengthchange_dLdt_annual_CMB_obs_ind_array_11_20
                    # r_cov = lengthchange_dLdt_annual_CMB_obs_ind_2_array_11_20


                    # generate the proposal (array)
                    if j == 0:
                        parameters_dict_noIndex = {key: value for key, value in parameters_dict.items() if key != "index"} 
                        # print("Original Dictionary Keys:", parameters_dict.keys())
                        # print("Updated Dictionary Keys:", parameters_dict_noIndex.keys())
                        # pdb.set_trace()
                        proposal_model = np.array(list(parameters_dict_noIndex.values())) # the proposal distributions of parameters used in the physical model
                    else:
                        proposal_model = thetap
                    #pdb.set_trace()
                    proposal = transform_space(proposal_model, 'to_normal') # TODO check the function glogit, how to deal with four parameters
                    
                    vars_to_perturbate = pygem_prms.vars_to_calibrate
                    priormean = np.zeros(len(vars_to_perturbate)) #TODO Should be the transformed values
                    priorsd = np.zeros(len(vars_to_perturbate))

                    for count, var in enumerate(vars_to_perturbate):
                        priormean[count] = pygem_prms.transformed_mean_priors[var]
                        priorsd[count] = pygem_prms.transformed_std_priors[var]
                        priorcov = np.diag(priorsd**2)
                    
                    #pdb.set_trace()
                    # interation
                    if j == 0:
                        
                        No = np.size(observations_sbst_masked) # TODO Check the dimention, here is the number of observations, should be 11, 10 for the length change and 1 for the mass balance
                        Ne = proposal.shape[1]
                        Nl = pygem_prms.max_iterations
                        Np = np.shape(proposal)[0]
                        predall = np.zeros([No, Ne, Nl])
                        predall[:] = np.nan
                        propsall = np.zeros([Np, Ne, Nl])
                        propsall[:] = np.nan
                        propmall = np.zeros([Np, Nl])
                        propmall[:] = np.nan
                        propsall_model = np.zeros([Np, Ne, Nl])
                        propsall_model[:] = np.nan
                        print('Np:', Np)
                        print('Ne:', Ne)
                        print('Nl:', Nl)
                        print('priormean :', priormean)
                        #pdb.set_trace()
                        propmall[:, j] = priormean
                        propcall = np.zeros([Np, Np, Nl])
                        propcall[:] = np.nan
                        propcall[:, :, j] = priorcov
                        adapt_thresh = pygem_prms.Neffthrs
                    #pdb.set_trace()
                    propsall[:, :, j] = proposal
                    # handle the failures in the model runs with nan values
                    n_expected = predall.shape[1]
                    n_actual = predicted.shape[1]

                    if n_actual != n_expected:
                        print(f"[WARNING] Sample {j} for glacier {rgiid_ind} has unexpected length: {n_actual} instead of {n_expected}")
                        predicted_padded = np.full((predicted.shape[0], n_expected), np.nan)
                        predicted_padded[:, :min(n_actual, n_expected)] = predicted[:, :min(n_actual, n_expected)]
                        predicted = predicted_padded
                    
                    predall[:, :, j] = predicted
                    ells = np.arange(j+1)
                    obs = observations_sbst_masked
                    # the proposall_model used in the model
                    propsall_model [:, :, j] = proposal_model
                    #priormean =priormean.reshape(-1,1)
                    #pdb.set_trace()
                    Weights_k, Neff_k = AMIS(obs, predall[:, :, ells],
                                    r_cov, priormean, priorcov,
                                    propmall[:, ells], propcall[:, :, ells],
                                    propsall[:, :, ells])

                    print('Neff: {Neff_k} in j:{j}'.format(Neff_k=int(Neff_k),j=j))
                    print('Weights_k is :', Weights_k)

                    diversity = Neff_k/Ne
                    doadapt = diversity < adapt_thresh
                    notlast = (j+1) < max_iterations
                    w = Weights_k.flatten('F') #TODO check the shape of the Weights_k, does it need to be flatten

                    # ==== save the parameters and Weights and Neff ====
                    # Dictionary to store dataset names and corresponding data arrays
                    output_data_dict_AMIS = {
                                            'Neff_k': Neff_k,
                                            'weights_array': Weights_k,
                                            'doadapt': doadapt,
                                            'notlast': notlast,
                                            'Iteration': j
                                            }
                    output_folder_AMIS = save_path_AMISINFO_glac #os.path.join(save_path_AMISINFO,glacier_str.split('.')[0].zfill(2)) # Assuming `pygem_prms.output_fp` exists
                    # Check if directory exists, otherwise create it
                    if not os.path.exists(output_folder_AMIS):
                        os.makedirs(output_folder_AMIS)
                    output_filename_AMIS = f"calibration_model_AMIS_Info_{rgiid_ind}_{j}.json" # dataset with weights and removed outliers compared to the prior samples/values
                    output_fp_AMIS = os.path.join(output_folder_AMIS, output_filename_AMIS)
                    # Save to JSON
                    with open(output_fp_AMIS, 'w') as f:
                        json.dump(output_data_dict_AMIS, f, indent=4, default=convert_to_serializable)

                    # ==== Can instead always set clip to 1 if you don't want to clip
                    doclip = doadapt and notlast
                    if doclip:
                        clip = int(np.round(adapt_thresh*Ne))
                        ws = -np.sort(-w)
                        wc = ws[clip-1]
                        nonzero = wc > 0
                        if nonzero:
                            toclip = w > wc
                            w[toclip] = wc
                            w = w/np.sum(w)
                        else:
                            doclip = False

                    Nw = np.size(w)
                    pinds = np.arange(Nw)
                    reinds = np.random.choice(pinds, Ne, p=w)
                    thetap = propsall[:, :, ells]
                    thetap = np.reshape(thetap, [Np, Nw], order='F')
                    thetap = thetap[:, reinds]
                    pm = np.mean(thetap, axis=1)
                    if doclip:
                        A = (thetap.T-pm).T
                        pc = (A@A.T)/Ne
                    else:
                        shrink=max(0.5**j,0.2) # TODO check the shrinkage factor
                        pc = np.copy(priorcov)*shrink
                    print("pc after AMIS is",pc)
                    # Draw from this Gaussian for the next adaptive iteration
                    # if there will be one
                    #pdb.set_trace()
                    if doadapt and notlast:

                        while True:
                            try:
                                L = np.linalg.cholesky(pc)
                                break
                            except np.linalg.LinAlgError:
                                pc = ct.cov_nearest(pc, method="clipped")
                                L = np.linalg.cholesky(pc)
                                print("np.linalg.LinAlgError in cholesky")
                                #pdb.set_trace()
                                break

                        Z = np.random.randn(Np, Ne)
                        thetap = (pm+(L@Z).T).T
                        propcall[:, :, j+1] = pc
                        propmall[:, j+1] = pm

                    # Update parameters for next iteration (it is just
                    # resampling if not adapt and/or last)

                    thetap = transform_space(thetap, 'from_normal')
                    # TODO how to update the ensembel
                    #Ensemble.iter_update(step, thetap, create=True, iteration=j)
                    #print("the thetap is ",thetap)
                    #pdb.set_trace()
                    # exit if not collapsed
                    if not doadapt:
                        break

                if (not doadapt) or (not notlast): # TODO add the information of the txt infor about good and maximum iterations, with the Neff
                    # Return the final ensemble and weights
                    param_prior_array_all = propsall_model[:,:,ells]
                    Np, Ne, Nl = np.shape(param_prior_array_all)
                    param_prior_array_all_reshape = np.reshape(param_prior_array_all,(Np, Ne*Nl),order = 'F')
                    param_post_array = param_prior_array_all_reshape[:, np.random.choice(Ne*Nl, size=Sample_N, replace=True, p=Weights_k)] # chose the replace true, means the weights is uniform 1/n
                    #transform the array to the parameters dictionary by adding the Index
                    parameters_dict_post = {key: value for key, value in zip(parameters_keys, param_post_array)}
                    parameters_dict_post['index'] = np.arange(Sample_N)
                    
                    #store the posterior of parameters
                    output_folder_params_poster = os.path.join(save_path_parameter_glac, 'Poster')
                    # Ensure directories exist
                    os.makedirs(output_folder_params_poster, exist_ok=True)
                    output_filename_params_poster = f'calibration_poster_Params_{rgiid_ind}.json'
                    output_fp_params = os.path.join(output_folder_params_poster, output_filename_params_poster)

                    # Save to JSON  
                    #pdb.set_trace()
                    with open(output_fp_params, "w") as f:
                        #json.dump(modelprms_data_serializable, f, indent=4)
                        json.dump(parameters_dict_post, f, indent=4, default=convert_to_serializable)

                    # save the statistics of the parameters
                    # Dictionary to store dataset names and corresponding data arrays
                    params_dict_post_statis = {"mean": np.mean(param_post_array, axis=1),
                                        "std": np.std(param_post_array, axis=1),
                                        "min": np.min(param_post_array, axis=1),
                                        "max": np.max(param_post_array, axis=1),
                                        "median": np.median(param_post_array, axis=1), 
                                        "IQR": np.percentile(param_post_array, 75, axis=1) - np.percentile(param_post_array, 25, axis=1),
                                        "MAD": median_abs_deviation(param_post_array, axis=1),
                                        "skewness": skew(param_post_array, axis=1),
                                        "kurtosis": kurtosis(param_post_array, axis=1)}
                    
                    output_folder_post_params_statis = os.path.join(save_path_parameter_glac)  # Assuming `pygem_prms.output_fp` exists
                    # Check if directory exists, otherwise create it
                    if not os.path.exists(output_folder_post_params_statis):
                        os.makedirs(output_folder_post_params_statis)
                    output_filename_post_params_statis = f"parameters_statistic_{rgiid_ind}_poster.json" # dataset with parameters
                    output_fp_post_params_statis = os.path.join(output_folder_post_params_statis, output_filename_post_params_statis)
                    # Save to JSON
                    with open(output_fp_post_params_statis, 'w') as f:
                        json.dump(params_dict_post_statis, f, indent=4, default=convert_to_serializable)

                    #%%
                    # ==== unique the posterior parameters, and save the unique parameters
                    param_post_array_unique,unique_indices,unique_counts = np.unique(param_post_array, axis = 1,return_index=True, return_counts=True)
                    param_post_array_unique_dic = {key: value for key, value in zip(parameters_keys, param_post_array_unique)}
                    param_post_array_unique_dic['index'] = np.arange(param_post_array_unique.shape[1]) 
                    output_folder_post_params_unique = os.path.join(save_path_parameter_glac,'Poster','Unique')
                    Weights_unique = unique_counts/np.sum(unique_counts)
                    # Ensure directories exist
                    os.makedirs(output_folder_post_params_unique, exist_ok=True)
                    output_filename_params_unique = f'calibration_poster_Params_unique_{rgiid_ind}.json'
                    output_fp_params_unique = os.path.join(output_folder_post_params_unique, output_filename_params_unique)
                    # Save to JSON
                    with open(output_fp_params_unique, "w") as f:
                        json.dump(param_post_array_unique_dic, f, indent=4, default=convert_to_serializable)

                    # ==== save the unique parameters, indices and the counts and weights in json
                    output_data_dict_unique_Info = {
                                            'parameters_array': param_post_array_unique, 
                                            'unique_indices': unique_indices,
                                            'unique_counts': unique_counts,
                                            'weights_array': Weights_unique
                                            }
                    output_filename_params_unique_Info = f'calibration_poster_Params_unique_{rgiid_ind}_Info.json'
                    output_fp_params_unique_Info = os.path.join(output_folder_post_params_unique, output_filename_params_unique_Info)                    
                    # Save to JSON
                    with open(output_fp_params_unique_Info, "w") as f:
                        json.dump(output_data_dict_unique_Info, f, indent=4, default=convert_to_serializable)
                    

                    #%%
                    # ==== recall the model to compute the model output based on the post parameters
                                        # Run the coupled model (PyGEM_OGGM_SERMeQ)
                    (lengthchange_dLdt_model_array_annual_post, lengthchange_m_TMS_model_array_annual_post, 
                    calving_flux_Gta_TMS_model_array_annual_post,massbalclim_TMS_model_array_annual_post,massbaltotal_TMS_model_array_annual_post,
                    massbalclim_TMS_model_array_annual_gta_post,massbaltotal_TMS_model_array_annual_gta_post,
                    FA_mwea_TMS_model_array_annual_post,area_km2_TMS_model_array_annual_post,velocity_at_calvingfront_model_array_annual_post,thickness_at_calvingfront_model_array_annual_post,
                    width_at_calvingfront_model_array_annual_post, volume_bsl_model_array_annual_post, 
                    volume_bwl_model_array_annual_post,calving_flux_Gta_average_model_array_post, 
                    calving_thickness_model_array_post, massbalclim_model_array_post,
                    massbaltotal_model_array_post, massbalclim_model_array_gta_post,
                    massbaltotal_model_array_gta_post,FA_mwea_average_model_array_post, 
                    mb_obs_mwea, mb_obs_mwea_err
                    ) = Model_MB_FA_RT(model_function=reg_calving_flux,
                                    parameters_dict=param_post_array_unique_dic,
                                    rgiid_ind=rgiid_ind,
                                    main_glac_rgi=main_glac_rgi_ind,
                                    fa_glac_data_reg=fa_glac_data_ind,
                                    ignore_nan=False,
                                    calibrate_timeseries=True,
                                    store_monthly_step=store_monthly_step,
                                    return_all=True,
                                    store_result=True,
                                    N_iteration="Poster",
                                    save_path_figure_glac=save_path_figure_glac,
                                    save_path_parameter_glac = save_path_parameter_glac,
                                    save_path_modeloutput_glac = save_path_modeloutput_glac,
                                    save_path_log_glac = save_path_log_glac,
                                    floating_info_fp_glac = floating_info_fp_glac,
                                    log_level = log_level
                                    )

                    # ==== get the average results # TODO check the dimention of the array, axis =1, or 0
                    #pdb.set_trace()

                    calving_flux_Gta_average_model_weighted = np.average(calving_flux_Gta_average_model_array_post.flatten(),weights = Weights_unique)
                    calving_thickness_model_weighted = np.average(calving_thickness_model_array_post.flatten(),weights = Weights_unique)
                    massbalclim_model_weighted = np.average(massbalclim_model_array_post.flatten(),weights = Weights_unique)
                    massbaltotal_model_weighted = np.average(massbaltotal_model_array_post.flatten(),weights = Weights_unique)
                    massbalclim_model_gta_weighted = np.average(massbalclim_model_array_gta_post.flatten(),weights = Weights_unique)
                    massbaltotal_model_gta_weighted = np.average(massbaltotal_model_array_gta_post.flatten(),weights = Weights_unique)
                    #pdb.set_trace()
                    FA_mwea_average_model_weighted = np.average(FA_mwea_average_model_array_post.flatten(),weights = Weights_unique)

                    lengthchange_dLdt_model_annual_weighted = np.average(lengthchange_dLdt_model_array_annual_post,axis = 1,weights = Weights_unique)
                    lengthchange_m_TMS_model_annual_weighted = np.average(lengthchange_m_TMS_model_array_annual_post,axis = 1,weights = Weights_unique)
                    calving_flux_Gta_TMS_model_annual_weighted = np.average(calving_flux_Gta_TMS_model_array_annual_post,axis = 1,weights = Weights_unique)
                    massbalclim_TMS_model_annual_weighted = np.average(massbalclim_TMS_model_array_annual_post,axis =1,weights = Weights_unique)
                    massbaltotal_TMS_model_annual_weighted = np.average(massbaltotal_TMS_model_array_annual_post,axis =1,weights = Weights_unique)
                    massbalclim_TMS_model_annual_gta_weighted = np.average(massbalclim_TMS_model_array_annual_gta_post,axis =1,weights = Weights_unique)
                    massbaltotal_TMS_model_annual_gta_weighted = np.average(massbaltotal_TMS_model_array_annual_gta_post,axis =1,weights = Weights_unique)
                    FA_mwea_TMS_model_annual_weighted = np.average(FA_mwea_TMS_model_array_annual_post,axis =1,weights = Weights_unique)
                    area_km2_TMS_model_annual_weighted = np.average(area_km2_TMS_model_array_annual_post,axis =1,weights = Weights_unique)   
                    velocity_at_calvingfront_model_array_annual_weighted = np.average(velocity_at_calvingfront_model_array_annual_post,axis =1,weights = Weights_unique)
                    thickness_at_calvingfront_model_array_annual_weighted = np.average(thickness_at_calvingfront_model_array_annual_post,axis =1,weights = Weights_unique)
                    width_at_calvingfront_model_array_annual_weighted = np.average(width_at_calvingfront_model_array_annual_post,axis =1,weights = Weights_unique)
                    volume_bsl_model_array_annual_weighted = np.average(volume_bsl_model_array_annual_post,axis =1,weights = Weights_unique)
                    volume_bwl_model_array_annual_weighted = np.average(volume_bwl_model_array_annual_post,axis =1,weights = Weights_unique)

                    # ==== Save the weighted results in json
                    output_folder_weights = os.path.join(save_path_modeloutput_glac,'Poster', 'Weighted')  # Assuming `pygem_prms.output_fp` exists
                    # Check if directory exists, otherwise create it
                    if not os.path.exists(output_folder_weights):
                        os.makedirs(output_folder_weights)
                    output_filename_weighted = f'calibration_weighted_output_{rgiid_ind}_poster.json' # dataset with weighted output
                    output_fp_weighted = os.path.join(output_folder_weights, output_filename_weighted)
                    # Dictionary to store dataset names and corresponding data arrays
                    dataset_dict_weighted = {'calving_flux_Gta_average_model_weighted': calving_flux_Gta_average_model_weighted,
                                            'calving_thickness_model_weighted_m': calving_thickness_model_weighted,
                                            'massbalclim_model_weighted_mwea': massbalclim_model_weighted,
                                            'massbaltotal_model_weighted_mwea': massbaltotal_model_weighted,
                                            'massbalclim_model_weighted_gta': massbalclim_model_gta_weighted,
                                            'massbaltotal_model_weighted_gta': massbaltotal_model_gta_weighted,
                                            'FA_mwea_average_model_weighted': FA_mwea_average_model_weighted,
                                            'lengthchange_dLdt_model_annual_weighted_myr': lengthchange_dLdt_model_annual_weighted,
                                            'lengthchange_m_TMS_model_annual_weighted': lengthchange_m_TMS_model_annual_weighted,
                                            'calving_flux_Gta_TMS_model_annual_weighted': calving_flux_Gta_TMS_model_annual_weighted,
                                            'massbalclim_TMS_model_annual_weighted_mwea': massbalclim_TMS_model_annual_weighted,
                                            'massbaltotal_TMS_model_annual_weighted_mwea': massbaltotal_TMS_model_annual_weighted,
                                            'massbalclim_TMS_model_annual_weighted_gta': massbalclim_TMS_model_annual_gta_weighted,
                                            'massbaltotal_TMS_model_annual_weighted_gta': massbaltotal_TMS_model_annual_gta_weighted,
                                            'FA_mwea_TMS_model_annual_weighted': FA_mwea_TMS_model_annual_weighted,
                                            'area_km2_TMS_model_annual_weighted': area_km2_TMS_model_annual_weighted,
                                            'velocity_at_calvingfront_model_array_annual_weighted_myr': velocity_at_calvingfront_model_array_annual_weighted,
                                            'thickness_at_calvingfront_model_array_annual_weighted_m': thickness_at_calvingfront_model_array_annual_weighted,
                                            'width_at_calvingfront_model_array_annual_weighted_m': width_at_calvingfront_model_array_annual_weighted,
                                            'volume_bsl_model_array_annual_weighted_m3': volume_bsl_model_array_annual_weighted,
                                            'volume_bwl_model_array_annual_weighted_m3': volume_bwl_model_array_annual_weighted}
                    # Save to json
                    with open(output_fp_weighted, 'w') as f:
                        json.dump(dataset_dict_weighted, f, indent=4, default=convert_to_serializable)

                    #pdb.set_trace()
                    # ==== Visulize the results
                    calving_flux_Gta_average_model_array_post_repeat = np.repeat(calving_flux_Gta_average_model_array_post,unique_counts,axis =0)
                    lengthchange_dLdt_model_array_annual_post_repeat = np.repeat(lengthchange_dLdt_model_array_annual_post,unique_counts,axis = 1)
                    lengthchange_m_TMS_model_array_annual_post_repeat = np.repeat(lengthchange_m_TMS_model_array_annual_post,unique_counts,axis = 1)
                    massbalclim_model_array_post_repeat = np.repeat(massbalclim_model_array_post,unique_counts,axis =0)
                    try:
                        # accoording to the unique counts, and the weights, recoverty the output array 
                        if Visualize_Index:

                            # priod avearge fa/calving_flux Gta  #TODO CHANGE TO THE POSTERIOR
                            # # Here is for debugging
                            # # Add debugging before line 2858:
                            # print(f"calving_flux_Gta_average_model_array_post_repeat shape: {np.shape(calving_flux_Gta_average_model_array_post_repeat)}")
                            # print(f"calving_flux_Gta_average_model_array_post_repeat type: {type(calving_flux_Gta_average_model_array_post_repeat)}")
                            # print(f"calving_flux_Gta_average_model_weighted shape: {np.shape(calving_flux_Gta_average_model_weighted)}")
                            # print(f"calving_flux_Gta_average_model_weighted type: {type(calving_flux_Gta_average_model_weighted)}")

                            # # Check if they're scalars or arrays
                            # if hasattr(calving_flux_Gta_average_model_weighted, '__len__'):
                            #     print(f"calving_flux_Gta_average_model_weighted length: {len(calving_flux_Gta_average_model_weighted)}")
                            # else:
                            #     print(f"calving_flux_Gta_average_model_weighted is scalar: {calving_flux_Gta_average_model_weighted}")

                            # print(f"fa_gta_obs_ind_array shape: {np.shape(fa_gta_obs_ind_array)}")
                            # print(f"fa_gta_obs_ind_array type: {type(fa_gta_obs_ind_array)}")
                            Visualization_timeseries.plot_model_vs_observation(np.concatenate([calving_flux_Gta_average_model_array_post_repeat,[[calving_flux_Gta_average_model_weighted]]], axis=0).tolist(),
                                                                            fa_gta_obs_ind_array, plot_type='point', model_label='Modeled frontal ablation (Gt a⁻¹)', obs_label='Observed frontal ablation (Gt a⁻¹)',
                                                                                model_legends=[f"Particle {i+1}" for i in range(Sample_N)] + ["Weighted Avg"], start_date=2000,
                                                                                title='Calving flux comparison model vs observation', observation_error=fa_gta_obs_unc_ind,
                                                                                save_path=save_path_figure_glac, save_name='Calving flux(20-year average) comparison model vs observation (weighted)')
                            
                            # length change rate dLdt vs observation
                            Visualization_timeseries.plot_model_vs_observation([*lengthchange_dLdt_model_array_annual_post_repeat.T, lengthchange_dLdt_model_annual_weighted], 
                                                                            lengthchange_dLdt_obs_ind, plot_type='timeseries',
                                                                            model_legends=[f"Particle {i+1}" for i in range(Sample_N)] + ["Weighted Avg"],
                                                                            title='Length Change Rate (dLdt) Comparison: Model vs Observation', xlabel='Year',
                                                                            ylabel='Length Change Rate (m a⁻¹)', observation_error=lengthchnage_dLdt_unc_obs_ind,
                                                                            start_date=2000, save_path=save_path_figure_glac, save_name='length_change_rate_comparison')
                            # length change m (the difference of the length of the elevation-band flowlines)vs observation
                            Visualization_timeseries.plot_model_vs_observation([*lengthchange_m_TMS_model_array_annual_post_repeat.T,
                                                                                lengthchange_m_TMS_model_annual_weighted], lengthchange_dLdt_obs_ind,
                                                                                plot_type='timeseries', model_legends=[f"Particle {i+1}" for i in range(Sample_N)] + ["Weighted Avg"],
                                                                                title='length change comparison model vs observation', xlabel='Year', ylabel='length change (m)',
                                                                                observation_error=lengthchnage_dLdt_unc_obs_ind, start_date=2000, save_path=save_path_figure_glac,
                                                                                save_name='length change comparison model vs observation (weighted)')

                            # massbalclim mwea vs observation #TODO at the moment model output is 20-year average, but the observation is 10-year average, should be changed later
                            # Visualization_timeseries.plot_model_vs_observation((np.append(massbalclim_model_array_post_repeat, massbalclim_model_weighted)).tolist(),
                            #                                                 mb_obs_mwea, plot_type='point', model_legends=[f"Particle {i+1}" for i in range(Sample_N)] + ["Weighted Avg"],
                            #                                                 title='mass balance climatology comparison model vs observation', xlabel='Year',
                            #                                                 ylabel='mass balance climatology (mwea)', observation_error=mb_obs_mwea_err, start_date=2000,
                            #                                                 save_path=save_path_figure_glac, save_name='mass balance climatology comparison model vs observation')
                    except:
                        print(traceback.format_exc())

                        # TODO add more choices if more observations are considered
                        # massbaltotal mwea vs observation
                        # frontal ablation mwea vs observation
                        # Velocity at the calving front vs observation
                        # Thickness at the calving front vs observation
                        # Width at the calving front vs observation
                        # Volume of the basal sliding zone vs observation
                        # Volume of the basal wetland zone vs observation
                        # massbalclim TMS mwea vs observation
                        # massbaltotal TMS mwea vs observation


                    # ==== save the monthly information in a hdf5 file and visulize the monthly information
                    if store_monthly_step:
                        # read the monthly posterior result and calculate the weighted average
                        output_folder_monthly = os.path.join(save_path_modeloutput_glac , 'Monthly')  # Assuming `pygem_prms.output_fp` exists
                        os.makedirs(output_folder_monthly, exist_ok=True)
                        output_filename_monthly = f"calibration_model_Monthly_output_{rgiid_ind}_Poster.json"  # Assuming `pygem_prms.output_fp` exists
                        output_fp_monthly = os.path.join(output_folder_monthly, output_filename_monthly)

                        # Read json
                        #pdb.set_trace()
                        with open(output_fp_monthly, 'r') as f:
                            output_data_dict_monthly = json.load(f)

                        # remove outliers based on the length change rate 
                        lengthchange_dLdt_model_array_monthly_post = np.array(output_data_dict_monthly['lengthchange_dLdt_model_array_monthly'])
                        lengthchange_m_TMS_model_array_monthly_post = np.array(output_data_dict_monthly['lengthchange_m_TMS_model_array_monthly'])
                        calving_flux_Gta_TMS_model_array_monthly_post = np.array(output_data_dict_monthly['calving_flux_Gta_TMS_model_array_monthly'])
                        FA_mwea_TMS_model_array_monthly_post = np.array(output_data_dict_monthly['FA_mwea_TMS_model_array_monthly'])
                        area_km2_TMS_model_array_monthly_post = np.array(output_data_dict_monthly['area_km2_TMS_model_array_monthly'])
                        velocity_at_calvingfront_model_array_monthly_post = np.array( output_data_dict_monthly['velocity_at_calvingfront_model_array_monthly'])
                        thickness_at_calvingfront_model_array_monthly_post = np.array(output_data_dict_monthly['thickness_at_calvingfront_model_array_monthly'])
                        width_at_calvingfront_model_array_monthly_post = np.array(output_data_dict_monthly['width_at_calvingfront_model_array_monthly'])
                        volume_bsl_model_array_monthly_post = np.array(output_data_dict_monthly['volume_bsl_model_array_monthly'])
                        volume_bwl_model_array_monthly_post = np.array(output_data_dict_monthly['volume_bwl_model_array_monthly'])
                        massbalclim_TMS_model_array_monthly_post = np.array(output_data_dict_monthly['massbalclim_TMS_model_array_monthly'])
                        massbaltotal_TMS_model_array_monthly_post =np.array( output_data_dict_monthly['massbaltotal_TMS_model_array_monthly'])
                        massbalclim_TMS_model_array_monthly_post_gta = np.array(output_data_dict_monthly['massbalclim_TMS_model_array_monthly_gta'])
                        massbaltotal_TMS_model_array_monthly_post_gta = np.array(output_data_dict_monthly['massbaltotal_TMS_model_array_monthly_gta'])

                        # === Weighted # TODO check the dimention, axis =0 or, axis = 1 ?
                        #pdb.set_trace()
                        lengthchange_dLdt_model_array_monthly_post_weighted = np.average(lengthchange_dLdt_model_array_monthly_post, axis=0,weights = Weights_unique)
                        lengthchange_m_TMS_model_array_monthly_post_weighted = np.average(lengthchange_m_TMS_model_array_monthly_post, axis=0,weights = Weights_unique)
                        calving_flux_Gta_TMS_model_array_monthly_post_weighted = np.average(calving_flux_Gta_TMS_model_array_monthly_post, axis=0,weights = Weights_unique)
                        FA_mwea_TMS_model_array_monthly_post_weighted = np.average(FA_mwea_TMS_model_array_monthly_post, axis=0,weights = Weights_unique)
                        area_km2_TMS_model_array_monthly_post_weighted = np.average(area_km2_TMS_model_array_monthly_post, axis=0,weights = Weights_unique)
                        velocity_at_calvingfront_model_array_monthly_post_weighted = np.average(velocity_at_calvingfront_model_array_monthly_post, axis=0,weights = Weights_unique)
                        thickness_at_calvingfront_model_array_monthly_post_weighted = np.average(thickness_at_calvingfront_model_array_monthly_post, axis=0,weights = Weights_unique)
                        width_at_calvingfront_model_array_monthly_post_weighted = np.average(width_at_calvingfront_model_array_monthly_post, axis=0,weights = Weights_unique)
                        volume_bsl_model_array_monthly_post_weighted = np.average(volume_bsl_model_array_monthly_post, axis=0,weights = Weights_unique)
                        volume_bwl_model_array_monthly_post_weighted = np.average(volume_bwl_model_array_monthly_post, axis=0,weights = Weights_unique)
                        massbalclim_TMS_model_array_monthly_post_weighted = np.average(np.asarray(massbalclim_TMS_model_array_monthly_post), axis=0,weights = Weights_unique)
                        massbaltotal_TMS_model_array_monthly_post_weighted = np.average(np.asarray(massbaltotal_TMS_model_array_monthly_post), axis=0,weights = Weights_unique)
                        massbalclim_TMS_model_array_monthly_post_gta_weighted = np.average(np.asarray(massbalclim_TMS_model_array_monthly_post_gta), axis=0,weights = Weights_unique)
                        massbaltotal_TMS_model_array_monthly_post_gta_weighted = np.average(np.asarray(massbaltotal_TMS_model_array_monthly_post_gta), axis=0,weights = Weights_unique)

                        # --- save the monthly information in a hdf5 file and visulize the monthly information
                        output_folder_monthly_weighted = os.path.join(save_path_modeloutput_glac, 'Poster','Weighted','Monthly')
                        os.makedirs(output_folder_monthly_weighted, exist_ok=True)
                        output_filename_monthly_weighted = f"calibration_model_Monthly_output_{rgiid_ind}_Poster_weighted.json"  # Assuming `pygem_prms.output_fp` exists
                        output_fp_monthly_weighted = os.path.join(output_folder_monthly_weighted, output_filename_monthly_weighted)
                        # Dictionary to store dataset names and corresponding data arrays
                        output_data_dict_monthly_weighted = {
                            'lengthchange_dLdt_model_array_monthly_post_weighted': lengthchange_dLdt_model_array_monthly_post_weighted,
                            'lengthchange_m_TMS_model_array_monthly_post_weighted': lengthchange_m_TMS_model_array_monthly_post_weighted,
                            'calving_flux_Gta_TMS_model_array_monthly_post_weighted': calving_flux_Gta_TMS_model_array_monthly_post_weighted,
                            'FA_mwea_TMS_model_array_monthly_post_weighted': FA_mwea_TMS_model_array_monthly_post_weighted,
                            'area_km2_TMS_model_array_monthly_post_weighted': area_km2_TMS_model_array_monthly_post_weighted,
                            'velocity_at_calvingfront_model_array_monthly_post_weighted': velocity_at_calvingfront_model_array_monthly_post_weighted,
                            'thickness_at_calvingfront_model_array_monthly_post_weighted': thickness_at_calvingfront_model_array_monthly_post_weighted,
                            'width_at_calvingfront_model_array_monthly_post_weighted': width_at_calvingfront_model_array_monthly_post_weighted,
                            'volume_bsl_model_array_monthly_post_weighted': volume_bsl_model_array_monthly_post_weighted,
                            'volume_bwl_model_array_monthly_post_weighted': volume_bwl_model_array_monthly_post_weighted,
                            'massbalclim_TMS_model_array_monthly_post_weighted': massbalclim_TMS_model_array_monthly_post_weighted,
                            'massbaltotal_TMS_model_array_monthly_post_weighted': massbaltotal_TMS_model_array_monthly_post_weighted,
                            'massbalclim_TMS_model_array_monthly_post_gta_weighted': massbalclim_TMS_model_array_monthly_post_gta_weighted,
                            'massbaltotal_TMS_model_array_monthly_post_gta_weighted': massbaltotal_TMS_model_array_monthly_post_gta_weighted
                        }
                        # Save to HDF5
                        with open(output_fp_monthly_weighted, 'w') as f:
                            json.dump(output_data_dict_monthly_weighted, f, indent=4, default=convert_to_serializable)

                        # --- visualize monthly information
                        Visualization_timeseries.plot_timeseries_Numpy(data = calving_flux_Gta_TMS_model_array_monthly_post_weighted*12, start_date='2000-01-01', end_date='2019-12-31',
                                                                    save_name='Timeseries of calving (monthly-weighted)',save_path=save_path_figure_glac, Y_label='calving flux (Gt/a)', F_title='Monthly Time Series-FA')
                        Visualization_timeseries.plot_timeseries_Numpy(data = lengthchange_dLdt_model_array_monthly_post_weighted, start_date='2000-01-01', end_date='2019-12-31',
                                                                    save_name='Timeseries of length change dLdt (monthly-weighted)',save_path=save_path_figure_glac, Y_label='length change rate (m a⁻¹)', F_title='Monthly Time Series-dLdt')
                        Visualization_timeseries.plot_timeseries_Numpy(data = velocity_at_calvingfront_model_array_monthly_post_weighted, start_date='2000-01-01', end_date='2019-12-31',
                                                                    save_name='Timeseries of velocity at calving front (monthly-weighted)',save_path=save_path_figure_glac, Y_label='velocity at calving front (m a⁻¹)', F_title='Monthly Time Series-velocity')
                        # more choice can be added in the future



                    # output as statistic information of the weighted results,#TODO add more information as users need
                    output_df_all.loc[nglac, 'calving_flux_Gta_average_model_weighted'] = calving_flux_Gta_average_model_weighted
                    output_df_all.loc[nglac, 'calving_thickness_model_weighted_m'] = calving_thickness_model_weighted
                    output_df_all.loc[nglac,'massbalclim_model_weighted_mwea'] = massbalclim_model_weighted
                    output_df_all.loc[nglac,'massbaltotal_model_weighted_mwea'] = massbaltotal_model_weighted
                    output_df_all.loc[nglac,'FA_mwea_average_model_weighted'] = FA_mwea_average_model_weighted
                    output_df_all.loc[nglac,'no_errors'] = 1
                    output_df_all.loc[nglac,'oggm_dynamics'] =1


                    # statistics of glaciers classes failed, good AMIS, or bad AMIS
                    if not doadapt:
                        output_df_all.loc[nglac,'good_AMIS'] = 1
                        output_df_all.loc[nglac,'bad_AMIS'] = 0
                        output_df_all.loc[nglac,'iterations'] = j
                        output_df_all.loc[nglac,'Neff_k'] = Neff_k
                        Good_AMIS.append(rgiid_ind)
                        N_good_AMIS += 1
                        N_good_iterations.append(j+1)

                    elif (not notlast):
                        # the iteration has reached the maximum but we didn't get the converged result
                        output_df_all.loc[nglac,'good_AMIS'] = 0
                        output_df_all.loc[nglac,'bad_AMIS'] = 1
                        output_df_all.loc[nglac,'iterations'] = j
                        output_df_all.loc[nglac,'Neff_k'] = Neff_k
                        Bad_AMIS.append(rgiid_ind)
                        N_bad_AMIS += 1 
                        N_bad_iterations.append(j+1)                      

                else :
                    # the AMIS failed, we didn't get the converged result
                    print("the AMIS failed, we didn't get the converged result.And the calibration falied and the Neff_k is:",Neff_k)
                    Failed_glacs.append(rgiid_ind)
                    N_failed += 1
                    pass
            except Exception as err:
                # Handle the exception and print the error message
                error_message = f"Error occurred for glacier {rgiid_ind}: {err}\n"
                traceback_info = traceback.format_exc()
                # Print the traceback information
                print(error_message)
                print(traceback_info)
                # Save the traceback information to a log file
                with open(os.path.join(save_path_log_glac, 'error_log.txt'), 'a') as log_file:
                    log_file.write("\n================\n")
                    log_file.write(error_message)
                    log_file.write(traceback_info)
                # Track failed glaciers
                Failed_glacs.append(rgiid_ind)
                N_failed += 1

        if len(Failed_glacs) > 0:
            failed_glaciers_fp = os.path.join(save_path_statistics_reg, 'Failed_glaciers.txt')
            file_exists_failed = os.path.exists(failed_glaciers_fp)
            mode_failed = 'a' if file_exists_failed else 'w'
            with open(failed_glaciers_fp, mode_failed) as f:
                if not file_exists_failed:
                    f.write("Failed Glaciers Log\n")
                    f.write("====================\n")
                f.write(f'There are {N_failed} glaciers that failed calibration\n')
                for glacier in Failed_glacs:
                    f.write(f"{glacier}\n")
                f.write("\n================\n")

        if Good_AMIS:
            good_amis_fp = os.path.join(save_path_statistics_reg, 'Good_AMIS.txt')
            file_exists_good= os.path.exists(good_amis_fp)
            mode_good = 'a' if file_exists_good else 'w'
            with open(good_amis_fp, mode_good) as f:
                if not file_exists_good:
                    f.write("Good AMIS Results Log\n")
                    f.write("====================\n")
                f.write(f'There are {N_good_AMIS} glaciers that passed calibration\n')
                for glacier, N_iters in zip(Good_AMIS, N_good_iterations):
                    f.write(f"{glacier}, with {N_iters} iterations\n")
                f.write("\n================\n")
        
        if Bad_AMIS:
            bad_amis_fp = os.path.join(save_path_statistics_reg, 'Bad_AMIS.txt')
            file_exists_bad = os.path.exists(bad_amis_fp)
            mode_bad = 'a' if file_exists_bad else 'w'
            with open(bad_amis_fp, mode_bad) as f:
                if not file_exists_bad:
                    f.write("Bad AMIS Results Log\n")
                    f.write("====================\n")
                f.write(f'There are {N_bad_AMIS} glaciers that failed calibration\n')
                for glacier, N_iters in zip(Bad_AMIS, N_bad_iterations):
                    f.write(f"{glacier}, with {N_iters} iterations\n")   
                f.write("\n================\n")
            
        #%% Save the output dataframe to a CSV file
        output_file_path = os.path.join(save_path_summary, output_fn)

        if not os.path.exists(output_file_path) or overwrite:
            print('Saving calibration results to:', output_file_path)
            output_df_all.to_csv(output_file_path, index=False)
            print('Calibration completed successfully')
        else:
            # Load the existing CSV if it exists and overwrite is False
            existing_df = pd.read_csv(output_file_path)

            # Append the new calibration results to the existing DataFrame
            output_df_all = pd.concat([existing_df, output_df_all], ignore_index=True)

            # Save the updated DataFrame back to the same file
            output_df_all.to_csv(output_file_path, index=False)
            print('Appended new results to existing calibration file.')



def main():
    """
    Model calibration
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the calibration output (specific output is dependent on the output option)
    """
    # Unpack variables
    args = getparser().parse_args()
    debug = args.debug == 1
    
    # Set up logging based on user input
    log_level = args.log_level
    # # get number of jobs #TODO For the parallel processing, need to be fixed
    # njobs = len(args.rgi_region01)
    # # number of cores for parallel processing
    # if args.ncores > 1:
    #     args.ncores = int(np.min([njobs, args.ncores]))

    #%% data paths =====
    # frontalabltion data (20-year averaged data)
    frontalablation_fp = pygem_prms.main_directory + '/../Calibration_dataset/frontalablation_average_data/'
    # frontalablation data (20-year averaged data) if available
    if args.frontalablation_fn is None:
        print("No frontalablation_fn provided. Please provide the proper frontalablation_fn.")
        exit(1)  # Exit the program with an error status
    else:
        frontalablation_fn = args.frontalablation_fn  # Assign the provided value

    # frontalablation data (annual data) if available
    frontalablation_annual_fp = pygem_prms.main_directory + '/../Calibration_dataset/frontalablation_annual_data/'
    frontalablation_annual_fn = args.frontalablation_annual_fn  # Assign the provided value
    # hugonnet data (climatic mass balance corrected by Frontal ablation) #TODO Check the dataset, which should be the climatic mass balance, not the total mass balance
    hugonnet_fp = pygem_prms.hugonnet_fp
    if args.hugonnet_fn is None:
        print("No hugonnet_fn provided. Please provide the proper hugonnet_fn.")
        exit(1)
    else:
        hugonnet_fn = args.hugonnet_fn
        pygem_prms.hugonnet_fn = hugonnet_fn
    # length change data
    lengthchange_annual_fp = pygem_prms.main_directory  + '/../Calibration_dataset/lengthchange_data/'
    if args.lengthchange_annual_fn is None:
        print("No lengthchange_annual_fn provided. Please provide the proper lengthchange_annual_fn.")
        exit(1)
    else:
        lengthchange_annual_fn = args.lengthchange_annual_fn
    # output data
    # base_dir = pygem_prms.main_directory
    # #output_fp = os.path.join(base_dir, '..', 'calving_data', 'analysis')
    # #output_fp = os.path.join(base_dir,'Calibration_AMIS_MB_FA_20102019')
    # output_fp = output_fp
    # os.makedirs(output_fp, exist_ok=True)

    #%% merge input claving datasets ===== #TODO At the moment, they are already merged, but need to be fixed for the future
    #merged_calving_data_fn = merge_data(frontalablation_fp=frontalablation_fp, overwrite=args.overwrite, verbose=args.verbose)

    #%% calibrate each individual glacier's parameters_Tbias_kp_ddfsnow_tau =====
    # call the function Cali_PBS_MB_FA_RT to calibrate the parameters
    cali_PBS_MB_FA_RT(regions = args.rgi_region01,
                       args = args,
                       frontalablation_fp = frontalablation_fp,
                       frontalablation_fn = frontalablation_fn,
                       output_fp = output_fp, 
                      hugonnet_fp = hugonnet_fp,
                      hugonnet_fn = hugonnet_fn,
                      lengthchange_annual_fp = lengthchange_annual_fp,
                      lengthchange_annual_fn = lengthchange_annual_fn,
                      overwrite=args.overwrite,
                      verbose=args.verbose,
                      Visualize_Index = args.Visualize_Index,
                      store_monthly_step = args.store_monthly_step,
                      debug = debug,
                      log_level = log_level) 
                      
                      #= args.Visualize_Index,store_monthly_step = args.store_monthly_step,debug = debug,debug_spc = debug_spc)

    # # calibrate each individual glacier's calving_k parameter #TODO Check the parallel processing
    # calib_ind_calving_k_partial = partial(calib_ind_calving_k, args=args, frontalablation_fp=frontalablation_fp, frontalablation_fn=merged_calving_data_fn, output_fp=output_fp, hugonnet2021_fp=hugonnet2021_fp)
    # with multiprocessing.Pool(args.ncores) as p:
    #     p.map(calib_ind_calving_k_partial, args.rgi_region01)

    #%% update the MB  #TODO Check the function for the update_mbdata
    # update reference mass balance data accordingly
    #update_mbdata(regions=args.rgi_region01, frontalablation_fp=output_fp, frontalablation_fn=frontalablation_cal_fn, hugonnet2021_fp=hugonnet2021_fp, hugonnet2021_facorr_fp=hugonnet2021_facorr_fp, ncores=args.ncores, overwrite=args.overwrite, verbose=args.verbose)

    #%%  plot results # TODO check the function for the plot_calving_k_allregions
    #plot_calving_k_allregions(output_fp=output_fp)


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    
    main()

    print('Total processing time:', time.time()-time_start, 's')