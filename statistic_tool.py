"""
Statistic Tool for analyzing and visualizing data.

This module provides a set of tools for statistical analysis, data visualization, and exporting results in various formats.

copyright: 2025, Ruitang Yang
author: ruitang.yang@geo.uio.no
version: 1.0.0

# This tool provides functions to calculate statistics, generate plots, and export results.
# For the presentday simulation/calibration, it includes functions to analyze the glacier length changes, mass balance, and other relevant metrics.
# It also supports exporting results to CSV and Excel formats for further analysis or reporting.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress


#%% Read the data (emsemble members and their corresponding info e.g. weights/counts for unique value) and calculate the statistics
def read_data_and_calculate_statistics(data_fp = None, unique_fp = None, unique_index = True,ensemble_members=None, weights=None):
    """
    Read data from a json file and calculate statistics for each ensemble member.

    Parameters:
    data_fp (str): File path to the file containing the data.
    unique_fp (str): File path to the file containing the weights for each ensemble member.
    unique_index (bool): If True, use the weights to restructure the DataFrame to ensure unique indices.
    ensemble_members (list): List of ensemble members to analyze.
    weights (list): List of weights corresponding to each ensemble member.

    Returns:
    pd.DataFrame: DataFrame containing the calculated statistics.
    """
    # Read the data (assuming it's in json/csv/xlsx format)
    if not os.path.exists(data_fp):
        raise FileNotFoundError(f"The file {data_fp} does not exist.")
    # Load the data into a DataFrame
    if data_fp.endswith('.json'):
        df = pd.read_json(data_fp)
    elif data_fp.endswith('.csv'):
        # If the data is in CSV format, read it using pandas
        if not os.path.exists(data_fp):
            raise FileNotFoundError(f"The file {data_fp} does not exist.")
        if data_fp.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(data_fp)
        else:
            raise ValueError("Unsupported file format. Please provide a .json or .csv file.")
    elif data_fp.endswith('.xlsx'):
        # If the data is in Excel format, read it using pandas
        if not os.path.exists(data_fp):
            raise FileNotFoundError(f"The file {data_fp} does not exist.")
        df = pd.read_excel(data_fp)
    else:
        raise ValueError("Unsupported file format. Please provide a .json, .csv, or .xlsx file.")
 
    # Check if the members are unique and weights/counts are provided, adn require restructuring
    if unique_index:
        if os.path.exists(unique_fp):
            # Load the weights/counts from the specified file
            if unique_fp.endswith('.json'):
                unique_info = pd.read_json(unique_fp)
            elif unique_fp.endswith('.csv'):
                unique_info = pd.read_csv(unique_fp)
            elif unique_fp.endswith('.xlsx'):
                unique_info = pd.read_excel(unique_fp)
            else:
                raise ValueError("Unsupported file format for weights. Please provide a .json, .csv, or .xlsx file.")
        else:
            raise FileNotFoundError(f"The file {unique_fp} does not exist.")
        
        # Check if the unique_info DataFrame contains the necessary columns
        if 'unique_counts' not in unique_info.columns or 'weight_array' not in unique_info.columns:
            raise ValueError("The unique_info DataFrame must contain 'member' and 'weight' columns.")
        else:
            # read the columns from the unique_info DataFrame about the counts/weights
            unique_counts = unique_info['unique_counts']
            #weight_array = unique_info['weight_array']
    
        # use the unique_info DataFrame to restructure the original DataFrame
        df = np.repeat(df.values, unique_counts.values, axis=1)
    else:
        # If unique_index is False, we assume the DataFrame is already structured correctly
        # and does not require any additional processing for unique indices.
        df = df.copy()

    #
    



    # If ensemble_members and weights are not provided, extract them from the DataFrame






    # Initialize a list to store results
    results = []

    # Calculate statistics for each ensemble member
    for member, weight in zip(ensemble_members, weights):
        member_data = df[df['member'] == member]
        mean_value = np.average(member_data['value'], weights=weight)
        std_dev = np.std(member_data['value'])
        results.append({'member': member, 'mean': mean_value, 'std_dev': std_dev})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df