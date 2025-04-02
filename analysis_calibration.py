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

# Load the local libraries
import pygem_input as pygem_prms

import Visualization_timeseries as Vis_ts



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
Vis_ts.plot_length_dl_TS_Annual(rgiid='RGI60-17.15808', output_path= '/Calibration_AMIS_MB_FA_20002010_N200/')


