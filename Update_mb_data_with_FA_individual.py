"""
Update the mass balance data (Geodetic) with the observed frontal ablation data for tidewater glaciers (individual glaciers) has observed frontal ablation data.
# It's a copy from the run_calibration_FA_Rt_New.py, with the option_update_mb_data = True, for set for individual glaciers.
@author: davidrounce/ Ruitang Yang 
"""
# Built-in libraries
#import argparse
import os
import pickle
import sys
import traceback
import shutil

# External libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import xarray as xr
import traceback
from scipy import special
import time
import ast  # To safely evaluate string representations of lists
import h5py
import pdb
from multiprocessing import Pool, cpu_count
from functools import partial
# Local libraries
try:
    import pygem
except:
    sys.path.append(os.getcwd() + '/../PyGEM/')
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory_with_calving
from pygem.shop import debris 
from pygem import class_climate
import Visualization_timeseries as Visualization_timeseries

import oggm
oggm_version = float(oggm.__version__[0:3])
from oggm import utils, cfg
from oggm import tasks
from oggm.core.flowline import FluxBasedModel
from oggm.core.calving_Ruitang import CalvingFluxBasedModelRt
from oggm.core.calving_Jan_Ruitang import CalvingFluxBasedModelJanRt
#from oggm.core.inversion import find_inversion_calving_from_any_mb
from oggm.core.inversion_RT_New import find_inversion_calving_from_any_mb
if oggm_version > 1.301:
    from oggm.core.massbalance import apparent_mb_from_any_mb # Newer Version of OGGM
else:
    from oggm.core.climate import apparent_mb_from_any_mb # Older Version of OGGM

# record the time
time_start = time.time()
# #%% ----- UPDATE MASS BALANCE DATA WITH FRONTAL ABLATION ESTIMATES  for indivdual glaciers with the observed frontal ablation-----

#%% ----- MANUAL INPUT DATA -----
#regions = [1,3,4,5,7,9,17,19]
regions = [17]

overwrite = True
    
# Load calving glacier data (already quality controlled during calibration)
#calving_fp = pygem_prms.main_directory + '/../calving_data/analysis_sermeq/'
calving_fp = pygem_prms.main_directory + '/../calving_data/'
calving_fn = 'frontalablation_data_test_'+pygem_prms.glac_no[0].split('.')[0]+'_'+ pygem_prms.glac_no[0].split('.')[1]+'.csv'
assert os.path.exists(calving_fp + calving_fn), 'Calibrated frontal ablation output dataset does not exist'
fa_glac_data = pd.read_csv(calving_fp + calving_fn)

# Load mass balance data
hugonnet_fn = 'df_pergla_global_20yr-filled.csv'
mb_data = pd.read_csv(pygem_prms.hugonnet_fp + hugonnet_fn)
mb_rgiids = list(mb_data.RGIId)

# Record prior data
mb_data['mb_romain_mwea'] = mb_data['mb_mwea']
mb_data['mb_romain_mwea_err'] = mb_data['mb_mwea_err']
mb_data['mb_clim_mwea'] = mb_data['mb_mwea']
mb_data['mb_clim_mwea_err'] = mb_data['mb_mwea_err']

# Update mass balance data
for nglac, rgiid in enumerate(fa_glac_data.RGIId):
    
    O1region = int(rgiid.split('-')[1].split('.')[0])
    if O1region in regions:        

        # Update the mass balance data in Romain's file
        mb_idx = mb_rgiids.index(rgiid)
        mb_data.loc[mb_idx,'mb_mwea'] = fa_glac_data.loc[nglac,'Romain_mwea_mbtot']
        mb_data.loc[mb_idx,'mb_clim_mwea'] = fa_glac_data.loc[nglac,'Romain_mwea_mbclim']
        
        print(rgiid, 'mb_mwea (the updated total mb):', np.round(mb_data.loc[mb_idx,'mb_mwea'],2), 
                'mb_clim (updated climate mb considering calving):', np.round(mb_data.loc[mb_idx,'mb_clim_mwea'],2), 
                'mb_romain (the initial mb obs):', np.round(mb_data.loc[mb_idx,'mb_romain_mwea'],2))

# Export the updated dataset
mb_data.to_csv(pygem_prms.hugonnet_fp + hugonnet_fn.replace('.csv','-facorrected.csv'), index=False)

# Update gdirs
#    rgiids = ['RGI60-' + x for x in pygem_prms.glac_no]
#    for nglac, rgiid in enumerate(rgiids):
for nglac, rgiid in enumerate(fa_glac_data.RGIId):
    
    O1region = int(rgiid.split('-')[1].split('.')[0])
    if O1region in regions:    

        print(rgiid)
    
        # Select subsets of data
        glacier_str = rgiid.split('-')[1]

        gdir = single_flowline_glacier_directory_with_calving(glacier_str, 
                                                                logging_level='CRITICAL',
                                                                reset=True
                                                                )