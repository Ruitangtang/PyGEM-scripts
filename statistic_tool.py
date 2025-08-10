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
import json
import ast
import re
from scipy.stats import ks_2samp  # Import K-S test function
import traceback
import pygem_input as pygem_prms


# Read data from CSV, Excel, or JSON files based on their extension
def read_data_from_file(file_path):
    """
    Read data from CSV, Excel, or JSON files based on their extension.
    
    Parameters:
    file_path (str): Path to the data file.
    Returns:
    pd.DataFrame: DataFrame containing the read data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Check the file extension and read the data accordingly
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


# Reconstruct the data for each glacier based on unique ensemble members and their corresponding counts
def reconstruct_data(data_dict, unique_counts):
    """
    Reconstruct the data for each glacier based on unique ensemble members and their corresponding counts.
    
    Parameters:
    data_dict (dict): Dictionary containing original data arrays for each key, where each value may be a 1D or 2D array.
    unique_counts (list or np.ndarray): Counts for each unique ensemble member.
    
    Returns:
    dict: A dictionary containing the reconstructed data with expanded entries according to the unique counts.
    """
    if unique_counts is None:
        raise ValueError("unique_counts must be provided to reconstruct the data.")
    else:
        # Ensure unique_counts is a NumPy array
        unique_counts = np.array(unique_counts)

    df_output = {}  # Initialize output dictionary

    for key, data_uniq in data_dict.items():
        data_uniq = np.array(data_uniq)  # Ensure data_uniq is a NumPy array
        
        if data_uniq.ndim == 1:
            # If data_uniq is a 1D array
            if len(unique_counts) != data_uniq.shape[0]:
                raise ValueError(f"Length of unique_counts must match the length of the 1D data_uniq for key: {key}.")
            data = np.repeat(data_uniq, unique_counts)
        
        elif data_uniq.ndim == 2:
            # If data_uniq is a 2D array
            rows = data_uniq.shape[0]  # Number of rows
            columns = data_uniq.shape[1]  # Number of columns

            if rows == len(unique_counts):
                # If the number of rows matches unique_counts, repeat along rows
                data = np.repeat(data_uniq, unique_counts, axis=0)
            elif columns == len(unique_counts):
                # If the number of columns matches unique_counts (must be 1 here), repeat along columns
                data = np.repeat(data_uniq, unique_counts, axis=1)
            else:
                raise ValueError(f"Dimensions of unique_counts must match one dimension of the 2D data_uniq for key: {key}. Expected counts to match rows {rows} or columns {columns}.")

        else:
            raise ValueError("Unsupported data dimension. Only 1D and 2D arrays are supported.")

        df_output[key] = data  # Store the reconstructed data in the output dictionary

    return df_output


# Compute fixed interval statistics （mean, sum） for a given dictionary of data arrays
def compute_fixed_interval_stats(data_dict, interval_years= 10, key_indices=None):
    """ Compute fixed interval statistics (mean, sum) for a given dictionary of data arrays.
    Parameters:
    data_dict (dict): Dictionary containing data arrays for each key, where each value is a 2D array (years x members).
    interval_years (int): Number of years for each fixed interval. The default is 10 years.
    key_indices (tuple, optional): Tuple of start and end indices to slice the keys from the dictionary. If None, all keys are used.
    Returns:
    means_results (dict): Dictionary containing mean values for each key, with shape [num_intervals, n_members].
    sums_results (dict): Dictionary containing sum values for each key, with shape [num_intervals, n_members].
    """
    means_results = {}
    sums_results = {}

    # Get keys from the dictionary
    keys = list(data_dict.keys())

    # Check if the key_indices parameter is provided
    if key_indices is not None:
        # Slice keys based on the provided indices
        start_index, end_index = key_indices
        keys = keys[start_index:end_index]

    for key in keys:
        data_array = data_dict[key]

        # Check that the data has two dimensions
        if data_array.ndim != 2:
            print(f"Data for key '{key}' does not have two dimensions. Actual dims: {data_array.ndim}")
            continue

        # Shape of data: expect (n_years, n_members)
        n_years, n_members = data_array.shape

        # Ensure that n_years is large enough for the specified intervals
        if n_years < interval_years:
            print(f"Data for key '{key}' does not have enough years ({n_years}) for the specified interval: {interval_years}.")
            continue

        # Calculate the number of complete intervals we can form
        num_intervals = n_years // interval_years
        
        # Initialize arrays to store means and sums for dynamic intervals
        means = np.zeros((num_intervals, n_members))
        sums = np.zeros((num_intervals, n_members))

        for i in range(num_intervals):
            start_idx = i * interval_years
            end_idx = start_idx + interval_years
            
            # Extract the interval data
            interval_data = data_array[start_idx:end_idx]
            
            # Calculate mean and sum for this interval
            means[i] = interval_data.mean(axis=0)  # Mean for each of the n_members
            sums[i] = interval_data.sum(axis=0)    # Sum for each of the n_members
        
        # Store the results in a dictionary
        means_results[key] = means  # Shape: [num_intervals, n_members]
        sums_results[key] = sums    # Shape: [num_intervals, n_members]

    return means_results, sums_results


# Process the data for each glacier
def read_extract_data_individual(file_path=None,file_name=None,unique_path=None,save_reconstructed=False):
    """ Process the data for each glacier, reconstruct depending on unique_counts,and save as new json (optional).
    
    Parameters:
    file_path (str): Path to the glacier data file.
    file_name (str): suffix of the file name, e.g. '.csv', '.xlsx', 'json',etc.
    unique_path (str): Path to the unique counts or weights file, if applicable.
    save_reconstructed (bool): If True, save the reconstructed data to a subfolder 'reconstructed_data'.
    
    Returns:
    pd.DataFrame: Processed DataFrame with relevant statistics.
    """
    # Example processing: Calculate mean and standard deviation of a specific column
    # Adjust this based on your actual data structure and requirements
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    else:
        # Read the data (assuming it's a CSV, Excel, or JSON file)
        glacier_data = read_data_from_file(file_path)

    # check and read the unique counts or weights from the specified file
    if not os.path.exists(unique_path):
        raise FileNotFoundError(f"The file {unique_path} does not exist.")
    else:
        # Read the unique counts or weights from the specified file
        unique_info = read_data_from_file(unique_path)
    print('')
    # read the columns from the unique_info DataFrame about the counts/weights
    unique_counts = unique_info['unique_counts']

    if glacier_data is None:
        raise ValueError(f"Failed to read data from {file_path}. Please check the file format and content.")
    else:
        # recontruct the data if necessary
        glacier_data_constructed = reconstruct_data(glacier_data, unique_counts)     
        # If save_reconstructed is True, save the reconstructed data to the specified folder
        if save_reconstructed:
            # save it under the subfolder 'reconstructed_data' in the glacier folder
            reconstructed_folder = os.path.join(os.path.dirname(file_path), 'reconstructed_data')
            if not os.path.exists(reconstructed_folder):
                os.makedirs(reconstructed_folder)
            # Ensure file_name is provided, if not, extract it from the file_path
            if file_name is None:
                file_name = os.path.basename(file_path)
            # Check if the file_name has an extension, if not, add '.json' as default
            if not file_name.endswith(('.json', '.csv', '.xlsx')):
                file_name += '.json'
            reconstructed_file_path = os.path.join(reconstructed_folder, file_name)
            if file_name.endswith('.json'):
                glacier_data_constructed.to_json(reconstructed_file_path, orient='records', lines=True)
            elif file_name.endswith('.csv'):
                glacier_data_constructed.to_csv(reconstructed_file_path, index=False)
            elif file_name.endswith('.xlsx'):
                glacier_data_constructed.to_excel(reconstructed_file_path, index=False)
            else:
                raise ValueError("Unsupported file format. Please provide a .json, .csv, or .xlsx file.")
        return glacier_data_constructed  # Return the processed DataFrame


# Split a dictionary of NumPy arrays into multiple dictionaries based on the input dimension
def split_dict_flexible(data_dict, split_dimension=1):
    """
    Splits a dictionary of NumPy arrays into multiple dictionaries based on the input dimension.
    
    Parameters:
    - data_dict (dict): A dictionary where each key maps to a NumPy array.
    - split_dimension (int): The number of rows to keep in each new dictionary.

    Returns:
    - split_dicts (list of dict): A list of dictionaries containing the split arrays.
    """
    # Check if split_dimension is valid
    if split_dimension <= 0:
        raise ValueError("split_dimension must be a positive integer.")
    
    split_dicts = []
    
    for key, value in data_dict.items():
        # Ensure the value is a NumPy array
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value for key '{key}' is not a NumPy array.")
        
        # Get the shape of the value
        n_rows, n_cols = value.shape
        
        # Ensure the number of rows is divisible by split_dimension
        if n_rows % split_dimension != 0:
            raise ValueError(f"Value for key '{key}' does not have a number of rows divisible by split_dimension. Found rows: {n_rows}")

        # Split the array into separate parts
        part_count = n_rows // split_dimension
        
        # Create empty dictionaries for each part
        for i in range(part_count):
            if i >= len(split_dicts):
                split_dicts.append({})  # Create a new dictionary if it doesn't exist
            
            # Slice the array into the new shape
            split_dicts[i][key] = value[i*split_dimension:(i+1)*split_dimension, :]
    
    return split_dicts


# Calculate the Highest Posterior Density Interval (HPDI)
def hdi(data, cred_mass=0.95):
    """Calculate the Highest Posterior Density Interval (HPDI).
    Parameters:
    data (np.ndarray): 1D array of data points.
    cred_mass (float): Credible mass for the HPDI, default is 0.95.
    Returns:
    tuple: A tuple containing the lower and upper bounds of the HPDI.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    hdi_index_inc = int(cred_mass * n)

    # List for potential intervals
    hdi_intervals = []

    for i in range(n - hdi_index_inc + 1):
        hdi_intervals.append((sorted_data[i], sorted_data[i + hdi_index_inc - 1]))

    # Find the interval with the smallest width
    hdi_widths = [(hdi[1] - hdi[0]) for hdi in hdi_intervals]
    min_width_index = np.argmin(hdi_widths)

    return hdi_intervals[min_width_index]


# Calculate statistics (mean, median, std, percentiles, mad, HPDI) for each key in a dictionary of data arrays
def calculate_statistics(data_dict):
    stats_results = {}
    
    for key, data_array in data_dict.items():
        # Ensure data is a NumPy array
        if not isinstance(data_array, np.ndarray):
            print(f"Data for key '{key}' is not a NumPy array.")
            continue
        
        # Ensure data is two-dimensional
        if data_array.ndim != 2:
            print(f"Data for key '{key}' does not have two dimensions. Actual dims: {data_array.ndim}")
            continue
        
        # Get the number of intervals and members
        n_intervals, n_members = data_array.shape
        
        # Initialize storage for statistics
        percentiles_results = np.zeros((n_intervals, 4))  # Shape for percentiles: [n_intervals, 4]
        hpdi_bounds = np.zeros((n_intervals, 2))  # 2 for lower and upper bounds of HPDI

        # Calculate statistics for each interval
        for i in range(n_intervals):
            interval_data = data_array[i, :]
            mean_value = np.mean(interval_data)  # Mean for interval
            median_value = np.median(interval_data)  # Median for interval
            std_value = np.std(interval_data)  # Standard deviation for interval
            mad_value = np.mean(np.abs(interval_data - mean_value))  # Calculate MAD
            
            # Store percentiles
            percentiles_results[i, :] = np.percentile(interval_data, [2.5, 25, 75, 97.5])
            hpdi_bounds[i, :] = hdi(interval_data)  # HPDI calculation
            
            # Store values directly into the results
            stats_results[key] = {
                'mean': mean_value,
                'median': median_value,
                'std': std_value,
                'percent_2.5': percentiles_results[i, 0],
                'percent_25': percentiles_results[i, 1],
                'percent_75': percentiles_results[i, 2],
                'percent_97.5': percentiles_results[i, 3],
                'mad': mad_value,
                'hdi_95_low': hpdi_bounds[i, 0],
                'hdi_95_high': hpdi_bounds[i, 1]
            }

    return stats_results


# Save a Pandas DataFrame to a CSV file
def save_dataframe_to_csv(file_path, file_name, data):
    """
    Save data to a CSV file. Converts to a Pandas DataFrame if necessary.

    Parameters:
    file_path (str): Directory where the CSV file will be saved.
    file_name (str): Name of the CSV file (without extension).
    data: Data to save (must be convertible to a DataFrame).

    Returns:
    None
    """
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"Failed to convert data to DataFrame: {e}")
            return

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Construct full file path
    full_path = os.path.join(file_path, f"{file_name}.csv")

    # Save the DataFrame
    data.to_csv(full_path, index=True)
    #print(f"DataFrame has been saved to {full_path}")
    return data


# calculate statistics from the dictionary of glacier data
def statistic_dict(glacier_data):
    """ 
    calculate the statistics for the dicts (mean,median,std, 2.5%,25%,75%,97.5%,mad, HPDI).
    
    Parameters:
    glacier_data (pd.DataFrame): DataFrame containing the glacier data to be processed.
    
    Returns:
    pd.DataFrame: Processed DataFrame with relevant statistics.
    """
    # Example processing: Calculate mean and standard deviation of a specific column
    # Adjust this based on your actual data structure and requirements
    processed_data = glacier_data.copy()
    
    # Assuming 'value' is a column in the DataFrame that we want to analyze
    if 'value' in processed_data.columns:
        processed_data['mean'] = processed_data['value'].mean()
        processed_data['std_dev'] = processed_data['value'].std()
    
    return processed_data


# Convert numpy arrays to lists for JSON serialization
def convert_to_serializable(obj):
    """Convert common non-serializable objects to serializable formats."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return str(obj)  # Fallback to string representation
    


#%% Read and extract data for all glaciers in a specified region
def read_extract_data_region(region_output_path = None, region_params_path= None,reg_id= None, data_index =None):
    """
    Read and extract data for all glaciers in a specified region, with RGI IDs included in outputs.

    Parameters:
    region_output_path (str): Path containing glacier folders with model results.
    region_params_path (str): Path to AMIS parameter folders for the region.
    reg_id (str): Identifier for the region, used for saving outputs. e.g. '07', '08','10' etc.
    data_index (str): Identifier for the data type ('Annual', 'Monthly', 'Poster', 'Prior').

    Returns:
    all_glacier_data_stats (list): List of dictionaries containing statistics for each glacier.
    all_glacier_data_mean (dict): Dictionary containing mean data for different periods.
    all_glacier_data_sum (dict): Dictionary containing sum data for different periods.

    """
    all_glacier_data_stats = []
    all_glacier_data_mean = {'2000_2010': [], '2010_2020': [], '2000_2020': []}
    all_glacier_data_sum = {'2000_2010': [], '2010_2020': [], '2000_2020': []}

    for glacier in os.listdir(region_output_path):
        glacier_path = os.path.join(region_output_path, glacier, data_index)
        info_path = os.path.join(region_params_path, glacier, 'Poster', 'Unique')

        if not (os.path.isdir(glacier_path) and data_index in ['Annual', 'Monthly']):
            print(f"Invalid or missing glacier path: {glacier_path}")
            continue

        for file in os.listdir(glacier_path):
            if not file.endswith('_Poster.json'):
                continue

            file_path = os.path.join(glacier_path, file)
            unique_file = next((u for u in os.listdir(info_path) if u.endswith('_Info.json')), None)
            unique_path = os.path.join(info_path, unique_file) if unique_file else None
            
            # Process data
            data = read_extract_data_individual(file_path=file_path, file_name=file, unique_path=unique_path)
            mean_10yrs, sum_10yrs = compute_fixed_interval_stats(data, interval_years=10, key_indices=(0, 14)) #TODO should be flexible as customize (0,11) for the first 11 keys, to exclude the multiyear averaged values
            mean_20yrs, sum_20yrs = compute_fixed_interval_stats(data, interval_years=20, key_indices=(0, 14))
            
            # Split stats
            mean_2000_2010, mean_2010_2020 = split_dict_flexible(mean_10yrs, split_dimension=1)
            sum_2000_2010, sum_2010_2020 = split_dict_flexible(sum_10yrs, split_dimension=1)

            # Create data records with RGI ID at top level
            rgi_id = glacier
            
            # Store mean data (flat structure)
            all_glacier_data_mean['2000_2010'].append({
                'rgiid': rgi_id,
                **mean_2000_2010  # Unpack all mean data fields
            })
            all_glacier_data_mean['2010_2020'].append({
                'rgiid': rgi_id,
                **mean_2010_2020
            })
            all_glacier_data_mean['2000_2020'].append({
                'rgiid': rgi_id,
                **mean_20yrs
            })
            
            # Store sum data (flat structure)
            all_glacier_data_sum['2000_2010'].append({
                'rgiid': rgi_id,
                **sum_2000_2010
            })
            all_glacier_data_sum['2010_2020'].append({
                'rgiid': rgi_id,
                **sum_2010_2020
            })
            all_glacier_data_sum['2000_2020'].append({
                'rgiid': rgi_id,
                **sum_20yrs
            })

            # Calculate and store statistics
            stats = {
                'rgiid': rgi_id,
                'mean_stats_20002010': calculate_statistics(mean_2000_2010),
                'mean_stats_20102020': calculate_statistics(mean_2010_2020),
                'mean_stats_20002020': calculate_statistics(mean_20yrs),
                'sum_stats_20002010': calculate_statistics(sum_2000_2010),
                'sum_stats_20102020': calculate_statistics(sum_2010_2020),
                'sum_stats_20002020': calculate_statistics(sum_20yrs),
            }
            all_glacier_data_stats.append(stats)

            # Save individual glacier stats
            stats_path = os.path.join(glacier_path, 'Statis_info')
            for name, df in [(k,v) for k,v in stats.items() if k != 'rgiid']:
                save_dataframe_to_csv(stats_path, f'{glacier}_{name}', df)
    # Sort all data by RGI ID
    def sort_by_rgiid(data_list):
        return sorted(data_list, key=lambda x: x['rgiid'])

    all_glacier_data_stats = sort_by_rgiid(all_glacier_data_stats)
    for period in all_glacier_data_mean:
        all_glacier_data_mean[period] = sort_by_rgiid(all_glacier_data_mean[period])
    for period in all_glacier_data_sum:
        all_glacier_data_sum[period] = sort_by_rgiid(all_glacier_data_sum[period])

    # Save regional data in the postprocessing folder
    region_output_path_Postprocessing = os.path.join(region_output_path,'..', '..','Postprocessing',reg_id, data_index)
    save_regional_data(region_output_path_Postprocessing, all_glacier_data_stats,all_glacier_data_mean,all_glacier_data_sum)

    return all_glacier_data_stats,all_glacier_data_mean,all_glacier_data_sum



# Save all regional data to appropriate files with proper serialization of numpy arrays
def save_regional_data(base_path,stats_list,mean_data, sum_data):
    """Save all regional data to appropriate files with proper serialization of numpy arrays.
    
    Parameters:
    base_path (str): Base path where the data will be saved.
    stats_list (list): List of dictionaries containing statistics for each glacier.
    mean_data (dict): Dictionary containing mean data for different periods.
    sum_data (dict): Dictionary containing sum data for different periods."""
    
    def convert_numpy(obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    # Prepare paths
    stats_path = os.path.join(base_path, 'Statis_info')
    data_path = os.path.join(base_path, 'Regional_raw_info')
    
    # Create directories if needed
    os.makedirs(stats_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # Convert and save statistics
    serializable_stats = [convert_to_serializable(stat) for stat in stats_list]
    serializable_stats.sort(key=lambda x: x['rgiid'])
    
    with open(os.path.join(stats_path, 'All_stats_Info_region.json'), 'w') as f:
        json.dump(serializable_stats, f, indent=4)

    # Convert and save mean and sum data
    for period in ['2000_2010', '2010_2020', '2000_2020']:
        # Convert numpy arrays in mean_data
        serializable_mean = convert_numpy(mean_data[period])
        with open(os.path.join(data_path, f'Regional_mean_{period}.json'), 'w') as f:
            json.dump(serializable_mean, f, indent=4)
        
        # Convert numpy arrays in sum_data
        serializable_sum = convert_numpy(sum_data[period])
        with open(os.path.join(data_path, f'Regional_sum_{period}.json'), 'w') as f:
            json.dump(serializable_sum, f, indent=4)



# the function to calculate the delta of the glacier length changes
def calculate_length_change(dLdt_data, interval_years=10):
    """
    Calculate glacier length changes and uncertainties over fixed intervals.

    Parameters:
    - dLdt_data (pd.DataFrame): Contains 'RGIId', 'dLdt_m_per_yr', 'dLdt_m_per_yr_unc' (as stringified lists).
    - interval_years (int): Years per interval (e.g., 10).

    Returns:
    - pd.DataFrame: With 'rgiid', 'length_change', 'length_change_unc', lists per glacier.
    """
    if not {'RGIId', 'dLdt_m_per_yr', 'dLdt_m_per_yr_unc'}.issubset(dLdt_data.columns):
        raise ValueError("Input must include 'RGIId', 'dLdt_m_per_yr', and 'dLdt_m_per_yr_unc'.")

    results = []

    for _, row in dLdt_data.iterrows():
        rgiid_raw = row['RGIId']
        rgiid = f"{int(rgiid_raw.split('-')[1].split('.')[0])}.{rgiid_raw.split('-')[1].split('.')[1]}"

        dLdt = np.array(ast.literal_eval(row['dLdt_m_per_yr']), dtype=float)
        dLdt_unc = np.array(ast.literal_eval(row['dLdt_m_per_yr_unc']), dtype=float)

        if len(dLdt) < interval_years:
            raise ValueError(f"Not enough data for {rgiid} to compute interval of {interval_years} years.")

        changes, uncertainties = [], []
        for i in range(len(dLdt) // interval_years):
            s, e = i * interval_years, (i + 1) * interval_years
            changes.append(np.sum(dLdt[s:e]))
            uncertainties.append(np.sqrt(np.sum(dLdt_unc[s:e] ** 2)))

        results.append({
            'rgiid': rgiid,
            'length_change': changes,
            'length_change_unc': uncertainties
        })

    return pd.DataFrame(results)


# Extract and save a DataFrame from a dictionary to a specified file format
def extract_and_save_df(dataframes_dict=None, df_name=None, output_path=None, file_name=None, file_format='csv',
                       data_index = 'Annual'):
    """
    Extracts a DataFrame from a dictionary and saves it to a specified file format.

    :param dataframes_dict: Dictionary containing DataFrames.
    :param df_name: Name of the DataFrame to extract.
    :param output_path: Path to save the DataFrame file (defaults to current directory if None).
    :param file_name: The name of the output file (without extension).
    :param file_format: Format to save as ('csv' or 'json'), defaults to 'csv'.
    :param data_index:  Identifier for the data type ('Annual', 'Monthly', 'Poster', 'Prior'), the default is 'Annual'.
    :return: The extracted DataFrame.
    """
    # Check if the DataFrame exists
    df = dataframes_dict.get(df_name)
    if df is None:
        raise ValueError(f"DataFrame '{df_name}' not found.")

    # Set output_path to current directory if None
    if output_path is None:
        output_path = os.getcwd()  # Use current working directory
        
    # Specify the path for saving the regional statistics info
    all_stats_path = os.path.join(output_path, data_index, 'Statis_info')

    # Create the output directory if it doesn't exist
    os.makedirs(all_stats_path, exist_ok=True)

    # Generate the full output file path
    file_path = os.path.join(all_stats_path, f"{file_name}.{file_format}")
    
    # Save DataFrame to the specified format
    if file_format == 'csv':
        df.to_csv(file_path, index=False)
    elif file_format == 'json':
        df.to_json(file_path, orient='records', lines=True)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
    
    #print(f"DataFrame '{df_name}' saved as {file_path}")
    return df


# Extract data from a nested structure and return a list of DataFrames
def extract_dataframes(data, primary_keys=None, key_dict=None, exclude_key='rgiid'):
    """
    Extracts data from the nested structure and returns a list of DataFrames.

    :param data: List of dictionaries containing keys with nested structure.
    :param primary_keys: List of primary keys to process (e.g., ['key_0', 'key_1']).
    :param key_dict: Specify which key's dictionary to process; defaults to item[key_name].
    :param exclude_key: Key to exclude from extraction (default is 'rgiid').
    :return: Dictionary of DataFrames, each associated with the respective key and a_x.
    """
    
    dataframes = {}

    # Use all keys if primary_keys is not defined
    if primary_keys is None:
        primary_keys = [key for key in data[0] if key != exclude_key]

    # Iterate over each item in the data list
    for item in data:
        rgiid = item.get(exclude_key)

        # Iterate over specified primary keys
        for key_name in primary_keys:
            # Use the provided key_dict or default to item[key_name]
            current_key_dict = key_dict if key_dict is not None else item[key_name]
            
            # Retrieve the a_x keys dynamically
            a_keys = [a_key for a_key in current_key_dict]

            # Create DataFrames for each a_x key
            for a_key in a_keys:
                # Prepare DataFrame data
                df_data = []
                
                # Accumulate b_x and rgiid
                for sub_item in data:
                    sub_rgiid = sub_item.get(exclude_key)
                    if key_name in sub_item:
                        sub_key_dict = sub_item[key_name]

                        if a_key in sub_key_dict:
                            b_values = {**sub_key_dict[a_key], exclude_key: sub_rgiid}
                            df_data.append(b_values)

                # Create a DataFrame and store it in the dictionary
                df = pd.DataFrame(df_data)
                # Name the DataFrame based on the key and a_key
                df_name = f"{key_name}_{a_key}"
                dataframes[df_name] = df

    return dataframes


#%% EXTRACT STATISTICS FROM A REGION
# result_dataframes = {}
# model_output_fp_region = None
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
# for df_name, file_name in stats_to_extract:
#     extract_and_save_df(dataframes_dict=result_dataframes,
#                         df_name=df_name,
#                         output_path=model_output_fp_region,
#                         file_name=file_name,
#                         file_format='csv')


#%% Extract and save ensemble data
def extract_and_save_ensemble_data(data_dict = None,save_path = None,data_index= 'Annual',
                                   key_value='lengthchange_dLdt_model_array_annual_myr',
                                   file_name= 'mean_dL_SQ_0020_raw_m',period= None,
                                   ensemble_func= None,file_format= 'csv',
                                   explode_ensembles = True):
    """
    Flexibly extract ensemble data and save to specified path.
    
    Args:
        data_dict: Dictionary containing glacier data (sum_data or mean_data)
        save_path: Directory to save output files
        data_index: Identifier for the data type ('Annual', 'Monthly', 'Poster', 'Prior')
        key_value: The data key to extract
        file_name: Base name for output files
        period: Which time period to extract (for mean_data), '2000_2010', '2010_2020', '2000_2020', or None for all
        ensemble_func: Function to apply to ensembles (e.g., np.mean)
        file_format: Output format ('csv', 'parquet', or 'feather')
        explode_ensembles: Whether to split ensemble members into columns
        
    Returns:
        DataFrame with extracted data and full save path if saved successfully
    """
    # Validate inputs
    if period and period not in data_dict:
        raise ValueError(f"Invalid period. Choose from: {list(data_dict.keys())}")
    
    if file_format not in ['csv', 'parquet', 'feather']:
        raise ValueError("file_format must be 'csv', 'parquet', or 'feather'")

    # Select data to process
    process_data = data_dict[period] if period else data_dict
    
    # Extract data
    extracted_data = []
    for glacier_data in process_data:
        rgiid = glacier_data['rgiid']
        ensemble_array = glacier_data.get(key_value)
        
        if ensemble_array is None:
            print(f"Warning: Key '{key_value}' not found for glacier {rgiid}")
            continue
            
        processed_data = ensemble_func(ensemble_array) if ensemble_func else ensemble_array
        
        extracted_data.append({
            'rgiid': rgiid,
            key_value: processed_data.tolist() if isinstance(processed_data, np.ndarray) else processed_data
        })

    # Create DataFrame
    df = pd.DataFrame(extracted_data)
    
    # Explode ensemble members if requested
    if explode_ensembles and not ensemble_func and isinstance(df[key_value].iloc[0], list):
        df = pd.concat([
            df['rgiid'], 
            pd.DataFrame(df[key_value].tolist())
        ], axis=1)
        df.columns = ['rgiid'] + [f'{key_value}' for i in range(len(df.columns)-1)]
    
    # Ensure save directory exists
    save_path = os.path.join(save_path,  data_index,'Regional_raw_info')
    os.makedirs(save_path, exist_ok=True)
    
    # Construct full file path
    save_file = f"{file_name}.{file_format}"
    full_path = os.path.join(save_path,save_file)
    
    # Save based on format
    try:
        if file_format == 'csv':
            df.to_csv(full_path, index=False)
        elif file_format == 'parquet':
            df.to_parquet(full_path)
        elif file_format == 'feather':
            df.to_feather(full_path)
        print(f"Successfully saved to {full_path}")
        return df
    except Exception as e:
        print(f"Error saving file: {e}")
        return df


# read the txt file and get the rgiid
def extract_rgi_ids(filepath = None,filename = None):
    """
    Extracts RGI IDs from a text file and formats them as [REGION].[GLACIER_ID] (e.g., 7.00029 from RGI60-07.00029).

    Args:
        filepath (str): Path to the directory containing the text file.
        filename (str): Path to the text file containing RGI IDs.

    Returns:
        pd.DataFrame: DataFrame with cleaned RGI IDs (columns: 'rgiid').
    """
    # The full file path and check if the file exists
    full_path = os.path.join(filepath, filename)
    if not os.path.exists(full_path):
        print(f"Warning: The file '{full_path}' does not exist.")  # Warning message
        return None  # Return None if the file does not exist

    with open(full_path, 'r') as file:
        content = file.read()

    # Improved regex to:
    # 1. Capture region (e.g., 07) and glacier ID (e.g., 00029)
    # 2. Handle optional leading zeros in the region
    matches = re.findall(r'RGI\d+-(\d{2})\.(\d{5})', content)

    # Process matches: 
    # - Remove leading zeros from region (e.g., 07 → 7)
    # - Keep all 5 digits of glacier ID (e.g., 00029 → 00029)
    rgiid_group = [
        f"{int(region)}.{glacier_id}"  # int() strips leading zeros
        for region, glacier_id in matches
    ]

    return pd.DataFrame(rgiid_group, columns=['rgiid'])


# Function to get the statistics information for the comparison of observed and modeled values
def get_statistics_compare(obs_change=None, obs_unc=None, hdi_high=None,hdi_low=None, model_mean=None,n_simulations=1000,
                           Loglevel = 'INFO',output_path = None,item_name = None,period = None,reg_id = None):
    """    Computes statistics for comparing observed and modeled values.
    Parameters:
    - obs_change (np.ndarray): Observed values.
    - obs_unc (np.ndarray): Uncertainty in observed values.
    - hdi_high (np.ndarray): High values of the highest density interval (HDI) for modeled values.
    - hdi_low (np.ndarray): Low values of the HDI for modeled values.
    - model_mean (np.ndarray): Mean of the modeled values.
    - n_simulations (int): Number of simulations for uncertainty adjustment. the default is 1000.
    - Loglevel (str): Logging level for output messages. Default is 'INFO'.
    - output_path (str): Path to save the output results. Default is None.
    - item_name (str): Name of the item being processed, used for logging. Default is None.
    - period (str): Time period for which the statistics are calculated, used for logging. Default is None.
    - reg_id (str): Region ID for which the statistics are calculated, used for logging. Default is None.
    Returns:
      dataframe with the following columns:

    - mean_d (float): Mean D statistic from K-S test.
    - mean_p (float): Mean p-value from K-S test.
    - nrmse (float): Normalized Root Mean Square Error.
    - coverage (float): Coverage of the HDI over the observed values.
    - overlap_hdi (float): Fraction of observations where the uncertainty range overlaps with the model
    - HDI_covs (np.ndarray): Boolean array indicating whether each observation is within the HDI.
    """
    # The output is a dataframe with the following columns:
    # mean_d, mean_p, nrmse, coverage, overlap_hdi, HDI_covs

    output_df = pd.DataFrame(columns=['mean_d', 'mean_p', 'nrmse', 'coverage', 'overlap_hdi', 'HDI_covs'])


    # Check if all inputs are provided
    if obs_change is None or obs_unc is None or hdi_high is None or hdi_low is None or model_mean is None:
        raise ValueError("All input arrays must be provided and not None.")
    # Check if all inputs are numpy arrays
    if not (isinstance(obs_change, np.ndarray) and isinstance(obs_unc, np.ndarray)
            and isinstance(hdi_high, np.ndarray) and isinstance(hdi_low, np.ndarray)
            and isinstance(model_mean, np.ndarray)):
        raise TypeError("All inputs must be numpy arrays.") 
    # Check if all input arrays have the same length
    if not (len(obs_change) == len(obs_unc) == len(hdi_high) == len(hdi_low) == len(model_mean)):
        raise ValueError("All input arrays must have the same length.") 
    # Ensure inputs are 1D arrays
    # --- (1) Compute K-S Test with Uncertainty ---
    d_values = []
    p_values = []
    
    for _ in range(n_simulations):
        # Perturb observations within uncertainty (Gaussian noise)
        obs_change = np.array(obs_change, dtype='float64')
        obs_unc = np.array(obs_unc, dtype='float64')

        perturbed_obs = obs_change + np.random.normal(0, obs_unc, size=len(obs_change))
        # Sample model predictions from HDI (uniform)
        perturbed_model = np.random.uniform(low=hdi_low, high=hdi_high, size=len(hdi_low))
        # K-S test
        d, p = ks_2samp(perturbed_obs, perturbed_model)
        d_values.append(d)
        p_values.append(p)
    
    mean_d = np.mean(d_values)
    mean_p = np.mean(p_values)
    
    # --- (2) Compute NRMSE ---
    nrmse = np.sqrt(np.mean((model_mean - obs_change)**2)) / np.mean(obs_unc)
    overlap_hdi = np.sum((obs_change - obs_unc <= hdi_high) & (obs_change + obs_unc >= hdi_low)) / len(obs_change)
    HDI_covs = (obs_change >= hdi_low) & (obs_change <= hdi_high)
    #print("the HDI covers:",HDI_covs)
    coverage = np.mean(HDI_covs)

    # Print the results
    if output_path is None:
        output_path = os.path.join(pygem_prms.output_filepath, 'Calibration','Postprocessing',reg_id,'Regional_analysis','Statistics_Info')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = output_path
    file_name = 'ks_test_results.txt'
    file_path_full = os.path.join(file_path, file_name)
    if Loglevel in ['INFO', 'DEBUG']:
        with open(file_path_full, 'a') as f:
            f.write(f"============================================ {item_name} and {period}============================================ \n")
            f.write(f"Uncertainty-Adjusted K-S Test:\nD = {mean_d:.3f}, p = {mean_p:.3f}\nNRMSE = {nrmse:.3f}\n")
            f.write(f"Observed 10th percentile: {np.percentile(obs_change, 10)}\n")
            f.write(f"Model 10th percentile: {np.percentile(model_mean, 10)}\n")
            f.write(f"Observed 90th percentile: {np.percentile(obs_change, 90)}\n")
            f.write(f"Model 90th percentile: {np.percentile(model_mean, 90)}\n")
            f.write(f"HDI coverage: {coverage:.1%}\n")
            f.write(f"Uncertainty-overlap coverage: {overlap_hdi:.1%}\n")


    # Store results in the output DataFrame
    output_df.loc[0] = [mean_d, mean_p, nrmse, coverage, overlap_hdi, HDI_covs]
    # Return the output DataFrame
    # Convert HDI coverage boolean array to a list for better readability
    HDI_covs = HDI_covs.tolist()
    output_df['HDI_covs'] = [HDI_covs]  # Store as a list in the DataFrame
    # Convert the output DataFrame to a tuple for return
    mean_d = output_df['mean_d'].values[0]
    mean_p = output_df['mean_p'].values[0]
    nrmse = output_df['nrmse'].values[0]
    coverage = output_df['coverage'].values[0]
    overlap_hdi = output_df['overlap_hdi'].values[0]
    HDI_covs = output_df['HDI_covs'].values[0]

    return output_df


# Function to clip lower errors to ensure values - err >= 0
def clip_errors(values = None,err = None,clip_lower= False):
    """Clip lower errors to ensure values - err >= 0.
    Parameters:
    - err (np.ndarray): Error values, can be symmetric or asymmetric.
    - values (np.ndarray): Corresponding values to which errors apply.
    - clip_lower (bool): If True, clip lower errors to ensure non-negativity.
    Returns:
    - np.ndarray: Clipped errors.
    """
    if err is None:
        return None
    err = np.asarray(err)
    if err.ndim == 2:  # Asymmetric errors [lower, upper]
        lower, upper = err
        lower_clipped = np.minimum(lower, values) if clip_lower else lower
        return [lower_clipped, upper]
    else:  # Symmetric errors
        return np.minimum(err, values) if clip_lower else err
    

# Function to read and load the observed data from csv files
def read_load_obs_unc(obs_data_fp=None):
    """Read and load observed data and uncertainties from a CSV file.
    
    Parameters:
    - obs_data_fp (str): Path to the CSV file containing observed data and uncertainties.
    
    Returns:
    - return a tube including all pd.df.
    """
    if not os.path.exists(obs_data_fp):
        raise FileNotFoundError(f"The file {obs_data_fp} does not exist.")
    
        # Define file paths and keys for data loading
    file_keys = {
        'fa_20002010': 'frontal_ablation_obs_20002010.csv',
        'fa_20102020': 'frontal_ablation_obs_20102020.csv',
        'fa_20002020': 'frontal_ablation_obs_20002020.csv',
        'dLdt_20002020': 'lengthchange_annual_rgi_region01_7_20002020.csv',
        'mb_20002010': 'mass_balance_obs_20002010.csv',
        'mb_20102020': 'mass_balance_obs_20102020.csv',
        'mb_20002020': 'mass_balance_obs_20002020.csv',
        'mb_20002020_corr': 'mass_balance_obs_20002020_corr.csv'
    }
    
    # Load the data using stats_t
    data_frames = {key: read_data_from_file(os.path.join(obs_data_fp, filename))
                   for key, filename in file_keys.items()}
    
    # Function to create DataFrames with common transformations
    def create_gta_df(obs_data):
        rgiid = [f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in obs_data.RGIId.values]
        return pd.DataFrame({
            'rgiid': rgiid,
            'fa_gta_obs': obs_data['fa_gta_obs'],
            'fa_gta_obs_unc': obs_data['fa_gta_obs_unc']
        })
    # Create frontal ablation DataFrames
    fa_obs_gta = {key: create_gta_df(data_frames[key]) for key in ['fa_20002010', 'fa_20102020', 'fa_20002020']}

    # Convert to m w.e. a-1 for frontal ablation
    def convert_fa_to_mwea(fa_df, area):
        return pd.DataFrame({
            'rgiid': fa_df['rgiid'],
            'fa_mwea_obs': fa_df['fa_gta_obs'] * 1000. / area,
            'fa_mwea_obs_unc': fa_df['fa_gta_obs_unc'] * 1000. / area
        })

    fa_obs_mwea = {key: convert_fa_to_mwea(fa_obs_gta[key], data_frames[key]['Area_km2']) for key in fa_obs_gta.keys()}

    # Function to create mass balance DataFrames
    def create_mwea_df(obs_data):
        rgiid = [f"{int(x.split('-')[1].split('.')[0])}.{x.split('-')[1].split('.')[1]}" for x in obs_data.RGIId.values]
        return pd.DataFrame({
            'rgiid': rgiid,
            'mb_clim_mwea': obs_data['mb_clim_mwea'],
            'mb_clim_mwea_err': obs_data['mb_clim_mwea_err']
        })
    # Create mass balance DataFrames
    mb_obs_mwea = {key: create_mwea_df(data_frames[key]) for key in ['mb_20002010', 'mb_20102020', 'mb_20002020', 'mb_20002020_corr']}

    # For mass balance convert to gt a-1
    def convert_mb_to_gta(mwea_df, observation_df):
        return pd.DataFrame({
            'rgiid': mwea_df['rgiid'],
            'mb_clim_gta': mwea_df['mb_clim_mwea'] / 1000. * observation_df['area'],
            'mb_clim_gta_err': mwea_df['mb_clim_mwea_err'] / 1000. * observation_df['area']
        })
    
    mb_obs_gta = {key: convert_mb_to_gta(mb_obs_mwea[key], data_frames[key]) for key in mb_obs_mwea.keys()}

    # Return the DataFrames directly
    return (fa_obs_gta['fa_20002010'], fa_obs_gta['fa_20102020'], fa_obs_gta['fa_20002020'],
            mb_obs_gta['mb_20002010'], mb_obs_gta['mb_20102020'], mb_obs_gta['mb_20002020'],
            fa_obs_mwea['fa_20002010'], fa_obs_mwea['fa_20102020'], fa_obs_mwea['fa_20002020'],
            mb_obs_mwea['mb_20002010'], mb_obs_mwea['mb_20102020'], mb_obs_mwea['mb_20002020'],
            mb_obs_mwea['mb_20002020_corr'], data_frames['dLdt_20002020'])


# Function to read the AMIS information and model output data for the prior simulation, to get the weighted prior model output of individual glaciers
def read_prior_compute_weighted_save_ind(AMIS_fp=None, model_output_fp=None,rgiid = None,save_path=None, data_index='Prior',
                            file_name= None, file_format='json'):
    """
    Reads AMIS information and model output data for the prior simulation, and saves the weighted prior model output.
    Parameters:
    - AMIS_fp (str): File path to the AMIS information file.
    - model_output_fp (str): File path to the model output data file.
    - rgiid (str): RGI ID for the glacier, used to format the output file name.
    - save_path (str): Directory to save the weighted prior model output.
    - data_index (str): Identifier for the data type ('Annual', 'Monthly', 'Poster', 'Prior').
    - file_name (str): Base name for the output file.
    - file_format (str): Format to save the output file ('json', 'csv', 'xlsx').
    Returns:
    - pd.DataFrame: DataFrame containing the weighted prior model output.
    """
    # == The full RGIID
    if rgiid is not None:
        # Ensure the RGI ID is formatted correctly, e.g., 'RGI60-07.00029', based on '7.00029', 'RGI60-10.00029', based on '10.00029
        RGIID = f"RGI60-{str(int(float(rgiid.split('.')[0]))).zfill(2)}.{rgiid.split('.')[1]}" # Remove 'RGI60-' prefix if present

    #== read the AMIS information file
    if not os.path.exists(AMIS_fp):
        raise FileNotFoundError(f"The file {AMIS_fp} does not exist.")
    AMIS_fn = 'calibration_model_AMIS_Info_' + RGIID + '_0.json'
    AMIS_fp_full = os.path.join(AMIS_fp, AMIS_fn)  # Assuming the AMIS info file is named 'AMIS_info.json'
    amis_info = pd.read_json(AMIS_fp_full)
    # get the weights
    weights = np.asarray(amis_info ['weights_array'])

    #== read the model output data file
    if not os.path.exists(model_output_fp):
        raise FileNotFoundError(f"The file {model_output_fp} does not exist.")
    if model_output_fp.endswith('_0.json'):
        with open(model_output_fp, 'r') as model_output_fp:
            model_output_data = json.load(model_output_fp)
    
    
    #print(" the type of model output data is:", type(model_output_data))
    
    #== compute the weighted prior model output
    prior_weighted = compute_weighted_average_and_save(
        data_dict=model_output_data,
        weights=weights,
        rgiid=rgiid,  # RGI ID is not provided in this context
        data_index=data_index,
        save_path=save_path,
        save_name=file_name,
        file_format=file_format
    ) 

    #== return the prior weighted model output
    return prior_weighted


# Function to compute the weighted average and save as json, return the Dictionary of weighted averages
def compute_weighted_average_and_save(data_dict=None, weights=None,rgiid =None, data_index = 'Prior',
                                       save_path=None,save_name=None, file_format='json'):
    """
    Computes the weighted average of a DataFrame and saves the result to a specified file format.
    
    Parameters:
    - data_dict (dict): Dictionary containing the data to compute the weighted average.
    - weights (list or np.ndarray): Weights for each row in the DataFrame.
    - rgiid (str): RGI ID for the glacier, used in the output file name.
    - data_index (str): Identifier for the data type ('Annual', 'Monthly', 'Poster', 'Prior').
    - save_path (str): Directory to save the output file.
    - save_name (str): Base name for the output file.
    - file_format (str): Format to save the output file ('json', 'csv', 'xlsx').
    
    Returns:
    - dictionary: Dict containing the weighted averages.
    """
    # Validate inputs
    if data_dict is None or weights is None:
        raise ValueError("data_df and weights must be provided.")
    
    if not isinstance(data_dict, dict):
        raise TypeError("data_df must be a dictionary.")
    
    if not isinstance(weights, (list, np.ndarray)):
        raise TypeError("weights must be a list or numpy array.")
    
    # Convert weights to a NumPy array if it isn't already
    weights = np.asarray(weights)
    
    # Dictionary to store weighted results
    dataset_dict_weighted = {}
    #print("the keys in the data_dict are:", data_dict.keys())
    # Iterate through the keys in the output_dict
    for key in data_dict.keys():
        data_array = data_dict[key]

        # Convert non-NumPy arrays to NumPy arrays
        if not isinstance(data_array, np.ndarray):
            data_array = np.asarray(data_array)

        # Check the shape of the array to decide on flattening/axis
        if data_array.ndim == 1:
            if len(weights) != data_array.size:
                raise ValueError(f"Weight size {len(weights)} does not match data array size {data_array.size} for key '{key}'.")
            weighted_average = np.average(data_array, weights=weights)
        elif data_array.ndim == 2:
            shape = data_array.shape
            
            if (shape[0] == 1 and len(weights) == shape[1]) or (shape[1] == 1 and len(weights) == shape[0]):
                # If one dimension is 1, weights must match the other dimension
                weighted_average = np.average(data_array, weights=weights, axis=1 if shape[0] == 1 else 0)
            elif shape[0] > 1 and shape[1] > 1:
                # If both dimensions are greater than 1, use shape[1] for averaging across columns
                if len(weights) != shape[1]:
                    raise ValueError(f"Weight size {len(weights)} does not match number of columns {shape[1]} in data array for key '{key}'.")
                weighted_average = np.average(data_array, axis=1, weights=weights)
            else:
                raise ValueError(f"Invalid conditions for weights and shapes for array with shape {shape} for key '{key}'.")
        else:
            print("Unexpected data array shape:", data_array.shape,"the key is : ",key,"data_array is:", data_array)
            raise ValueError(f"Unexpected array shape for key '{key}': {data_array.shape}")

        # Store the weighted average result in the dictionary
        weighted_key = f"{key}_weighted"
        dataset_dict_weighted[weighted_key] = weighted_average

    # outputpath
    if save_path is None:
        print("No save path provided, using current working directory.")
        save_path = os.getcwd()  # Use current working directory if no path is provided
    else:
        save_path = os.path.join(save_path,rgiid,data_index,'Weighted')# Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)
    # Construct the full file name
    if rgiid is not None:
        # Ensure the RGI ID is formatted correctly, e.g., 'RGI60-07.00029', based on '7.00029', 'RGI60-10.00029', based on '10.00029
        rgiid = f"RGI60-{str(int(float(rgiid.split('.')[0]))).zfill(2)}.{rgiid.split('.')[1]}" # Remove 'RGI60-' prefix if present
    if save_name is None:
        save_name = 'calibration_weighted_output_'+ rgiid+ f"_{data_index}".lower()
    file_name = f"{save_name}.{file_format}"
    file_full_path = os.path.join(save_path, file_name)
     
    # Save the weighted averages to a file
    if file_format == 'json':
        import json
        # Convert the dictionary to a JSON serializable format
        dataset_dict_weighted = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in dataset_dict_weighted.items()}
        # Save to a JSON file
        with open(file_full_path, 'w') as json_file:
            json.dump(dataset_dict_weighted, json_file, indent=4)
        #print(f"Weighted averages saved to {file_full_path}")
        return dataset_dict_weighted
    elif file_format == 'csv':
        # Convert the dictionary to a DataFrame and save as CSV
        dataset_df = pd.DataFrame(dataset_dict_weighted)
        dataset_df.to_csv(file_full_path, index=False)
        #print(f"Weighted averages saved to {file_full_path}")
        return dataset_df
    elif file_format == 'xlsx':
        # Convert the dictionary to a DataFrame and save as Excel
        dataset_df = pd.DataFrame(dataset_dict_weighted)
        dataset_df.to_excel(file_full_path, index=False)
        #print(f"Weighted averages saved to {file_full_path}")
        return dataset_df
    else:
        raise ValueError("Unsupported file format. Please use 'json', 'csv', or 'xlsx'.")


# Function to read the AMIS information and model output data for the prior simulation, to get the weighted prior model output of regional glaciers
def read_prior_compute_weighted_save_region(model_output_region_fp=None, AMIS_fp = None, data_index='Annual',
                                            file_name=None, file_format='json'):
    """    Reads AMIS information and model output data for the prior simulation, and saves the weighted prior model output.
    returns a dictionary of prior weighted model output for each glacier in the region, for all the keys and the key 'rgiid')
    Parameters:
    - model_output_fp (str): File path to the model output data file.
    - AMIS_fp (str): File path to the AMIS information file.
    - data_index (str): Identifier for the data type ('Annual', 'Monthly', 'Poster', 'Prior').
    - file_name (str): Base name for the output file.
    - file_format (str): Format to save the output file ('json', 'csv', 'xlsx').
    returns:
    - Prior_weighted_allghted (dict): Dictionary containing the weighted prior model output for each glacier in the region.

    """
    # return  the weighted prior model output of regional glaciers
    Prior_weighted_all = {'rgiid': [],}

    for glacier in os.listdir(model_output_region_fp):
        glacier_path = os.path.join(model_output_region_fp, glacier, data_index)
        amis_path = os.path.join(AMIS_fp, glacier)

        if not (os.path.isdir(glacier_path) and data_index in ['Annual', 'Poster']):
            print(f"Invalid or missing glacier path: {glacier_path}")
            continue
        rgiid = glacier
        #print("rgiid is:", rgiid)
        # print("amis_path is:", amis_path) 
        for file in os.listdir(glacier_path):
            if not file.endswith('_0.json'):
                continue

            file_path = os.path.join(glacier_path, file)

            # Read the prior model output and compute the weighted average
            prior_weighted = read_prior_compute_weighted_save_ind(
                AMIS_fp=amis_path,
                model_output_fp=file_path,
                rgiid=rgiid,
                save_path=model_output_region_fp,
                data_index='Prior',
                file_name=file_name,
                file_format=file_format
            )
            # add the rgiid to the prior weighted model output
            Prior_weighted_all['rgiid'].append(rgiid)   
            # Iterate over the keys in each glacier data, excluding 'rgiid'
            for key, value in prior_weighted.items():
                if key != 'rgiid':  # Exclude the 'rgiid' since it's already handled
                    if key not in Prior_weighted_all:
                        Prior_weighted_all[key] = []  # Initialize the list if the key doesn't exist
                    Prior_weighted_all[key].append(value)  # Append the value to the corresponding list
    # Return the dictionary of prior weighted model output
    return Prior_weighted_all


# Function to calculate the RMSE between observed and modeled values, with optional uncertainty adjustment
def calculate_rmse_with_unc(obs_values=None, model_values=None, obs_uncertainty=None, adjust_uncertainty=False):
    """    Calculates the Root Mean Square Error (RMSE) between observed and modeled values, with optional uncertainty adjustment.
    Parameters:
    - obs_values (np.ndarray): Observed values.
    - model_values (np.ndarray): Modeled values.
    - obs_uncertainty (np.ndarray): Uncertainty in observed values, used for adjustment.
    - adjust_uncertainty (bool): If True, adjusts the RMSE calculation using the observed uncertainty.
    Returns:
    - float: The calculated RMSE value.
    """
    # Check if all inputs are provided
    if obs_values is None or model_values is None:
        raise ValueError("Both observed and modeled values must be provided.")
    
    # Check if all inputs are numpy arrays
    if not (isinstance(obs_values, np.ndarray) and isinstance(model_values, np.ndarray)):
        raise TypeError("Observed and modeled values must be numpy arrays.")
    
    # Ensure the input arrays have the same length
    if len(obs_values) != len(model_values):
        raise ValueError("Observed and modeled values must have the same length.")
    
    # If uncertainty adjustment is requested, check if uncertainty is provided
    if adjust_uncertainty:
        if obs_uncertainty is None:
            raise ValueError("Observed uncertainty must be provided for adjustment.")
        if not isinstance(obs_uncertainty, np.ndarray):
            raise TypeError("Observed uncertainty must be a numpy array.")
        if len(obs_uncertainty) != len(obs_values):
            raise ValueError("Observed uncertainty must have the same length as observed values.")

        # Adjust the RMSE calculation using the observed uncertainty
        # filter out zero uncertainties to avoid division by zero
        valid_indices = obs_uncertainty > 0
        obs_values = obs_values[valid_indices]
        model_values = model_values[valid_indices]
        obs_uncertainty = obs_uncertainty[valid_indices]
        try:
            rmse = np.sqrt(np.mean(((obs_values - model_values) / obs_uncertainty) ** 2))
        except ValueError:
            rmse = np.nan  # Assign NaN if a value error occurs
            print(traceback.format_exc())
        except ZeroDivisionError:
            rmse = np.nan  # Assign NaN if division by zero occurs
            print(traceback.format_exc())
    else:
        # Calculate RMSE without uncertainty adjustment
        rmse = np.sqrt(np.mean((obs_values - model_values) ** 2))
    
    return rmse


# Function to load the weighted posterior model output for regional glaciers
def load_posterior_weighted_region(model_output_fp_region=None):
    """    Loads the weighted posterior model output for regional glaciers.
    Parameters:
    - model_output_fp_region (str): File path to the model output data for the region
    Returns:
    - Poster_weighted (dict): Dictionary containing the weighted posterior model output for each glacier in the region.
    """

    # Initialize the weighted posterior model output dictionary
    Poster_weighted = {'rgiid': []}

    # Iterate over each glacier directory in the specified path
    for glacier in os.listdir(model_output_fp_region):
        # Read the posterior weighted file path
        glacier_path = os.path.join(model_output_fp_region, glacier, 'Poster', 'Weighted')
        rgiid = glacier
        RGIID = f"RGI60-{str(int(float(rgiid.split('.')[0]))).zfill(2)}.{rgiid.split('.')[1]}"
        poster_file_name = f'calibration_weighted_output_{RGIID}_poster.json'  # use lowercase for filename
        output_file_path_poster = os.path.join(glacier_path, poster_file_name)

        # Check if the posterior file exists
        if not os.path.exists(output_file_path_poster):
            print(f"File {output_file_path_poster} does not exist. Continuing to the next glacier.")
            continue  # Skip to the next iteration if the file does not exist

        # Load the JSON file
        with open(output_file_path_poster, 'r') as f:
            output_poster = json.load(f)

        # Add the rgiid to the Poster_weighted dictionary
        Poster_weighted['rgiid'].append(rgiid)

        # Iterate over the keys in each glacier data, excluding 'rgiid'
        for key, value in output_poster.items():
            if key != 'rgiid':  # Exclude the 'rgiid' since it's already handled
                if key not in Poster_weighted:
                    Poster_weighted[key] = []  # Initialize the list if the key doesn't exist
                Poster_weighted[key].append(value)  # Append the value to the corresponding list
    
    return Poster_weighted

# Custom warning handler defined outside the function
def custom_warning_handler(message, category, filename, lineno, file=None, line=None,
                           object_name=None, file_fp='warning_record.txt'):
    """
    Writes warning messages to a specified log file.

    Parameters
    ----------
    message : str
        The warning message.
    category : Warning
        The warning category (e.g., RuntimeWarning, UserWarning).
    filename : str
        The name of the file where the warning was raised.
    lineno : int
        The line number of the warning.
    file : TextIO, optional
        The file object (default is None).
    line : str, optional
        The line of code (default is None).
    object_name : str, optional
        The name of the object associated with the warning (default is None).
    file_fp : str, optional
        The file path for the warning information (default is 'warning_record.txt').
    """
    with open(file_fp, "a") as wf:  # Open the file in append mode
        if object_name is not None:
            wf.write(f"Object Name: {object_name}\n")  # Log object name on the first line
        wf.write(f"{filename}:{lineno}: {category.__name__}: {message}\n")  # Log the warning details


# Log error (traceback info)
def log_traceback(object_name=None, traceback_fp='traceback.txt'):
    """
    Logs traceback messages to a specified error log file.

    Parameters
    ----------
    object_name : str, optional
        The name of the object associated with the error (default is None).
    file_fp : str, optional
        The file path for the error information (default is 'traceback.txt').
    """
    with open(traceback_fp, "a") as ef:  # Open the file in append mode
        if object_name is not None:
            ef.write(f"Object Name: {object_name}\n")  # Log object name on the first line
        ef.write(f"TRACEBACK: {traceback.format_exc()}\n")  # Log the traceback information
