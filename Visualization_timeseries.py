## Plotting timeseries variables for each glacier
## Ruitang Yang (ruitang.yang@geo.uio.no)
## Last update: 2024-08-29


# Required libraries
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import ast
import math
import matplotlib
from matplotlib.gridspec import GridSpec
# matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
import os
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D  # Import Line2D for custom legend
import pdb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.lines as mlines  # Import for custom legend handling

import json


import pygem_input as pygem_prms
# Load the data




# function to plot timeseries of numpy data
def plot_timeseries_Numpy(data, start_date='2000-01-01', end_date='2020-12-31', save_name=None,
                          save_path=None, Y_label=None, F_title='Time Series'):
    """
    Plots a time series of variable with dashed grid lines after each year.

    Parameters:
    - data (numpy array): The variable data.
    - start_date (str) The starting date of the time series in 'YYYY-MM-DD' format.
    - end_date (str) The ending date of the time series in 'YYYY-MM-DD' format.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    - Y_label (str or None): The label for the y-axis. If None, the y-axis label will be 'Variable Name'.
    - F_title (str or None): The title for the figure. If None, the figure title will be 'Time Series'.
    """

    # Create the datetime index with monthly frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')

    # Check if data length matches date range
    if data is None or len(data) != len(date_range):
        raise ValueError("Data length must match the number of time points in the date range.")
    # Format the datetime index to 'YYYY-MM' for tick labels
    # date_range_Tick = [d.strftime('%Y-%m') for d in date_range]

    
    # Plotting the data with the new datetime index
    plt.figure(figsize=(12, 6))
    plt.plot(date_range, data , color='blue')
    
    # Set x-axis ticks for January and July of each year, and x,y labels
    major_ticks = date_range[(date_range.month == 7)]  # Get every January and July
    plt.xticks(major_ticks)  # Set ticks at major_ticks
    # Create labels for ticks
    tick_labels = []
    for date in major_ticks:
        month_str = date.strftime('%b')  # Get month abbreviation (Jan or Jul)
        year_str = date.strftime('%Y')    # Get year
        if date.year % 2 == 0:  # Only label every 2 years
            tick_labels.append(f"{month_str}\n{year_str}")  # Format to two lines
        else:
            tick_labels.append("")  # Empty label for non-5-year marks
    plt.gca().set_xticklabels(tick_labels, rotation=0, ha='center')  # Set tick labels


    plt.xlabel('Date')
    plt.ylabel(Y_label if Y_label else 'Variable Name')

    # Adding labels and title
    plt.title(F_title)

    # Adding dashed vertical lines after each year
    years = pd.date_range(start=date_range.min(), end=date_range.max(), freq='Y')  # Year start frequency
    for year in years:
        plt.axvline(x=year, linestyle='--', color='gray', linewidth=0.5)  # Add a dashed vertical line

    # Set x-axis limits based on the min and max of the date_range
    plt.xlim(date_range.min(), date_range.max())    

    # Show legend
    plt.legend()

    # Save the figure if a save path is provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")

    # Display the plot
    #plt.show()
    


# function to plot timeseries of list data
def plot_timeseries_List(data = None, start_year=2020,ylabel= None,xlabel= None,
                         title= None, save_path=None,save_name =None):
    """
    Plots a time series of variable with dashed grid lines after each year.
    Parameters:
    - data (list): The variable data.
    - start_year (int) The starting year of the time series.
    - ylabel (str or None): The label for the y-axis. If None, the y-axis label will be 'Variable Name'.
    - xlabel (str or None): The label for the x-axis. If None, the x-axis label will be 'Date'.
    - title (str or None): The title for the figure. If None, the figure title will be 'Time Series'.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.
    """
    # Generate a list of years based on the length of the data
    years = list(range(start_year, start_year + len(data)))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(years, data, marker='*', linestyle='-', color='k')
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Show the grid and the plot
    #plt.grid(True)
    #plt.show()
    
    # Save the figure if save_path is provided
    if save_path and save_name:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")


# function to plot timeseries of xarray data
def plot_timeseries (calving_m3, base_year=2000, save_name = None, save_path = None):
    """
    Plots a time series of calving flux with dashed grid lines after each year.

    Parameters:
    - calving_m3 (xarray.DataArray): The calving flux data with coordinates 'calendar_year' and 'calendar_month'.
    - base_year (int): The base year corresponding to calendar_year = 0. Default is 2000.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    """
    # Create a datetime index combining calendar_year and calendar_month
    calendar_year = calving_m3.coords['calendar_year'].values
    calendar_month = calving_m3.coords['calendar_month'].values

    # Create datetime index
    dates = pd.to_datetime({
        'year': base_year + calendar_year,
        'month': calendar_month,
        'day': 1  # Set all to the first day of the month
    })

    # Plotting the data with the new datetime index
    plt.figure(figsize=(12, 6))
    plt.plot(dates, calving_m3, label='Calving Flux (m³)', color='blue')

    # Adding labels and title
    #plt.title('Calving Flux Time Series')
    plt.xlabel('Date')
    plt.ylabel('Calving Flux (m³)')
    #plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Basic grid for both axes

    # Adding dashed vertical lines after each year
    years = pd.date_range(start=dates.min(), end=dates.max(), freq='YS')  # Year start frequency
    for year in years:
        plt.axvline(x=year, linestyle='--', color='gray', linewidth=0.5)  # Add a dashed vertical line

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show legend
    plt.legend()

    # Save the figure if a save path is provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")

    # Display the plot
    #plt.show()




# Function to create a timeseries for each glacier in each month, for the centerline flowing
def plot_timeseries_profile(gdir, filesuffix ='', sel_years = None, group='fl_0', ax=None, ylabel='Elevation (m a.s.l.)',
                         title='Flowline profile', save_path=None,save_name =None,xlabel='Distance along the flowline (km)'):
    """
    Plots elevation bands from a NetCDF dataset using xarray and optionally saves the figure.

    Parameters:
    - gdir (GlacierDirectory): The glacier directory object containing the dataset.
    - filesuffix (str): The file suffix identifier for the specific diagnostics file.
    - sel_years (list or array-like): The years to select for plotting the thickness.
    - group (str, optional): The group within the NetCDF file to read data from. Default is 'fl_0', for the centerline flowline (elevation-band)
    - ax (matplotlib.axes._axes.Axes, optional): The axes to plot on. If None, a new figure and axes will be created.
    - ylabel (str, optional): The label for the y-axis. Default is 'Elevation (m a.s.l.)'.
    - title (str, optional): The title of the plot. Default is 'Flowline profile'.
    - save_path (str, optional): The full path and filename to save the figure. If None, the figure will not be saved.
    - x_label (str, optional): The label for the x-axis. Default is 'Distance along the flowline (m)'.

    Returns:
    - matplotlib.axes._axes.Axes: The axes with the plot.
    """
    # Open the dataset
    with xr.open_dataset(gdir.get_filepath('fl_diagnostics', filesuffix=filesuffix), group=group) as ds:
        # Create a new figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))

        # if the selected years are not provided, plot all years
        if sel_years is None:
            sel_years = ds.time

        # get the distance along the flowline
        distance = ds.dis_along_flowline/1000

        # Get the water level
        WL = ds.attrs['water_level']

        # Add a dashed line at y=0 and y=  water_level
        if WL ==0 :
            ax.axhline(y=WL, color='gray', linestyle='--', linewidth=1)
        else:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.axhline(y=WL, color='r', linestyle='--', linewidth=1)

        # Plot bed elevation as a baseline
        ax.plot(distance, ds.bed_h, color='black', label='Bed elevation')

        # Generate a color palette from Seaborn
        colors = sns.color_palette('rocket', len(sel_years))  


        # Plot each year with the gradient colors
        # # Plot the bed height plus thickness for the selected years
        # (ds.bed_h + ds.sel(time=sel_years).thickness_m).plot(ax=ax, hue='time')
        for i, year in enumerate(sel_years):
            ax.plot(distance, ds.bed_h + ds.sel(time=year).thickness_m, color=colors[i], label=str(year.values))

  
        # Create a custom legend with gradient colors and only year labels
        custom_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(sel_years))]

        if WL ==0:
            legend_labels = ['Bed elevation'] + [str(int(year.values)+2000) for year in sel_years]
            custom_lines.insert(0, Line2D([0], [0], color='black', lw=2))  # Add the bed elevation line
        else:
            legend_labels = ['Water level','Bed elevation'] + [str(int(year.values)+2000) for year in sel_years]
            custom_lines.insert(0, Line2D([0], [0], color='r', lw=2, linestyle = '--'))  # Add the water level line
            custom_lines.insert(1, Line2D([0], [0], color='black', lw=2))  # Add the bed elevation line
            
        # Display the legend
        legend = ax.legend(custom_lines, legend_labels, loc='upper right')
        
        # Set labels and title
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)

        # Save the figure if save_path is provided
        if save_path:
        # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path_full = os.path.join(save_path, save_name)
            plt.savefig(save_path_full, bbox_inches='tight')
            print(f"Figure saved to {save_path_full}")

    return ax



# Function to plot timeseries snapshot of glacier variables
def plot_time_series_snapshots(gdir, filesuffix='', sel_times = None, n_year =1,variable='thickness_m', group='fl_0', 
                               ylabel='Variable Value', xlabel='Distance along the flowline (m)', title='Time Series Snapshots', 
                               save_path=None,save_name=None):
    """
    Plots snapshots of a time series variable from a NetCDF dataset using xarray.

    Parameters:
    - gdir (GlacierDirectory): The glacier directory object containing the dataset.
    - filesuffix (str): The file suffix identifier for the specific diagnostics file.
    - sel_times (list or array-like): The time points to select for plotting the variable.
    - n_year (int) : the number of year interval to show, Default is 1
    - variable (str, optional): The variable name in the dataset to plot. Default is 'thickness_m'.
    - group (str, optional): The group within the NetCDF file to read data from. Default is 'fl_0'.
    - ylabel (str, optional): The label for the y-axis. Default is 'Variable Value'.
    - xlabel (str, optional): The label for the x-axis. Default is 'Distance (m)'.
    - title (str, optional): The base title of the plots. Each plot will have the time appended to the title.
    - save_path (str, optional): The directory path to save the figure. If None, the figure will not be saved.

    Returns:
    - None: Displays the plots and optionally saves them.
    """
    # Open the dataset
    with xr.open_dataset(gdir.get_filepath('fl_diagnostics', filesuffix=filesuffix), group=group) as ds:
        # generate the sel_titimes,(list or array-like): The time points to select for plotting the variable.
        # start_date = datetime.strptime(start_date, '%Y-%m-%d')
        # end_date = datetime.strptime(end_date, '%Y-%m-%d')
        # sel_times = []
        # current_date = start_date
        # while current_date <= end_date:
        #     sel_times.append(current_date.strftime('%Y-%m-%d'))
        #     # Move to the next year
        #     next_year = current_date.year + n_year
        #     current_date = current_date.replace(year=next_year)
        if sel_times is None:
            sel_times = ds.time.values
        else:
            sel_times = sel_times
        sel_times = sel_times[::n_year]

        # get the distance along the flowline
        distance = ds.dis_along_flowline/1000

        # Get the water level
        WL = ds.attrs['water_level']

        # Set up a grid of subplots
        num_snapshots = len(sel_times)
        nrows = 7
        ncols = int(num_snapshots / 7) if num_snapshots % 7 == 0 else (num_snapshots // 7) + 1  # Calculate number of columns
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(num_snapshots, 6*nrows), sharey=True)
        # Flatten the axes array to make it easier to iterate
        axes = axes.ravel()  # Convert to 1D array

        # Loop through selected time points and plot snapshots
        for i, time_point in enumerate(sel_times):
            # Select data at the specified time
            data_at_time = ds.sel(time=time_point)[variable]

            # Plot on the corresponding axis
            ax = axes[i] if num_snapshots > 1 else axes

            # add dashed lines at y=0 and y=water_level
            if WL ==0 :
                ax.axhline(y=WL, color='gray', linestyle='--', linewidth=1)
            else:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
                ax.axhline(y=WL, color='r', linestyle='--', linewidth=1)
        
            # Plot bed elevation and variable at the selected time
            ax.plot(distance, ds.bed_h, color='black', label='Bed elevation')
            ax.plot(distance, ds.bed_h+data_at_time, color='blue', label=f'Time: {time_point+2000}', marker='*', markersize =4)
            
            # Create a custom legend with gradient colors and only year labels
            custom_lines = [Line2D([0], [0], color='blue', lw=2)]
            if WL ==0:
                legend_labels = ['Bed elevation'] + ['Surface elevation']
                custom_lines.insert(0, Line2D([0], [0], color='black', lw=2))  # Add the bed elevation line
            else:
                legend_labels = ['Water level','Bed elevation'] + ['Surface elevation']
                custom_lines.insert(0, Line2D([0], [0], color='r', lw=2, linestyle = '--'))  # Add the water level line
                custom_lines.insert(1, Line2D([0], [0], color='black', lw=2))  # Add the bed elevation line
            
            # Display the legend
            legend = ax.legend(custom_lines, legend_labels, loc='upper right')
            # Set labels and title
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(f'{title} - {time_point+2000}')


        # Adjust layout
        plt.tight_layout()

        # Save the figure if save_path is provided
        if save_path:
        # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path_full = os.path.join(save_path, save_name)
            plt.savefig(save_path_full, bbox_inches='tight')
            print(f"Figure saved to {save_path_full}")

        # Show the plot
        #plt.show()
# Example usage with synthetic data
# selected_times = ['2000-01-01', '2010-01-01', '2020-01-01']  # Define the times to plot
# plot_time_series_snapshots(ds, selected_times)



# Function to plot an animation of timeseries glacier variables

def animate_time_series(gdir, filesuffix ='',variable='thickness_m', group='fl_0',interval=400, ylabel='Elevation (m a.s.l.)', 
                        xlabel='Distance along the flowline (km)', title='Elevation Changes Animate', save_path=None,save_name=None):
    """
    Creates an animation of a time series variable from a NetCDF dataset using xarray.

    Parameters:
    - gdir (GlacierDirectory): The glacier directory object containing the dataset.
    - filesuffix (str): The file suffix identifier for the specific diagnostics file.
    - variable (str, optional): The variable name in the dataset to animate. Default is 'thickness_m'.
    - group (str, optional): The group within the NetCDF file to read data from. Default is 'fl_0'.
    - interval (int, optional): Delay between frames in milliseconds. Default is 200.
    - ylabel (str, optional): The label for the y-axis. Default is 'Elevation (m a.s.l.)'.
    - xlabel (str, optional): The label for the x-axis. Default is 'Distance (m)'.
    - title (str, optional): The title of the animation plot.
    - save_path (str, optional): The file path to save the animation. If None, the animation will not be saved.

    Returns:
    - None: Displays the animation and optionally saves it.
    """

    # Open the dataset
    with xr.open_dataset(gdir.get_filepath('fl_diagnostics', filesuffix=filesuffix), group=group) as ds: 
        # ds (xarray Dataset): The dataset containing the variable to plot.


        # Extract the time points and distances, water_level
        # times = ds['time'].values
        # distance = ds['distance'].values
        times = ds.time
        distance = ds.dis_along_flowline/1000
        WL = ds.attrs['water_level']
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'b-', marker='*',markersize =4)
        
        
        # Set axis labels and title
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        
        # Initialize the plot limits
        ax.set_xlim(distance.min(), distance.max())
        ax.set_ylim((ds[variable]+ds.bed_h).min(), math.ceil((ds[variable]+ds.bed_h).max()/500)*500)

        # Add the horizontal dashed line at y = 0 and y= water_level
        if WL ==0 :
            ax.axhline(y=WL, color='gray', linestyle='--', linewidth=1)
        else:
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.axhline(y=WL, color='r', linestyle='--', linewidth=1)
            
        # Plot the bed elevation as a dashed line
        bed_line, = ax.plot(distance, ds.bed_h, 'k-', label='Bed Elevation')


        
        def update(frame):
            # Update the line data for the current frame (time point)
            time_point = times[frame]
            data_at_time = ds.sel(time=time_point)[variable]
            line.set_data(distance, data_at_time+ds.bed_h)
            # Convert the string to a NumPy datetime64 object
            time_point_np = np.datetime64(f"{int(2000 + time_point)}-01-01")
            ax.set_title(f'{title} - {np.datetime_as_string(time_point_np, unit="Y")}')
            return line,

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)

        # Save the animation if a save_path is provided
        if save_path:
        # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path_full_gif = os.path.join(save_path, save_name +'.gif')
            save_path_full_mp4 = os.path.join(save_path, save_name +'.mp4')
            anim.save(save_path_full_gif, writer='pillow')
            #anim.save(save_path_full_mp4, writer='ffmpeg')
            print(f"Animation saved to {save_path_full_gif}")
            #print(f"Animation saved to {save_path_full_mp4}")

        # Show the animation
        #plt.show()

# # Example usage with synthetic data
# times = pd.date_range('2000-01-01', '2020-01-01', freq='YS')
# distance = np.linspace(0, 1000, 50)  # Simulated distance array
# elevation_data = np.random.randn(len(times), len(distance)) * 20 + 500

# # Create a dataset
# ds = xr.Dataset(
#     {
#         "thickness_m": (["time", "distance"], elevation_data),
#     },
#     coords={
#         "time": times,
#         "distance": distance
#     }
# )

# # Call the function to create and show the animation
# animate_time_series(ds, save_path='time_series_animation.gif')
            

# Function to plot the comparison of the model output and observations
def plot_model_vs_observation(models_output, observation_data, dates=None, start_date = None,plot_type='point', 
                              model_legends= None, model_label ='Model Output', obs_label='Observation', 
                              title='Model vs Observation Comparison', ylabel=None, xlabel='Time/Points', 
                              save_path=None,save_name=None,observation_error=None,save_name_legend=None):
    """
    Plots the comparison between model output and observation data.
    
    Parameters:
    - models_output:A list of lists/arrays, where each sublist/array contains model output values.
    - observation_data: A list or array of observation values.
    - dates: Optional list of dates for timeseries comparison (should match the length of model_output/observation_data).
    - start_date: Optional start date for timeseries comparison. e.g. 2000
    - plot_type: 'point' for scatter plot (point-to-point comparison), 'timeseries' for timeseries comparison.
    - model_legends: A list of labels for each model in the models_output list.
    - model_label: Label for the model data in the plot.
    - obs_label: Label for the observation data in the plot.
    - title: Title of the plot.
    - ylabel: Label for the y-axis.
    - xlabel: Label for the x-axis.
    - save_path (str, optional): The file path to save the figure. If None, the figure will not be saved.
    - save_name (str, optional): The file name to save the figure. If None, the figure will not be saved.
    - save_name_legend (str, optional): The file name to save the legend. If None, the legend will not be saved.
    - observation_error: Optional list or array of error values associated with the observation data.
 
    Returns:
    - None: Displays the plot and optionally saves it.
    """

    if model_legends is None:
        # Generate default labels if not provided
        model_legends = [f"Model {i+1}" for i in range(len(models_output))]
            # Plot each model's output against the observation data
    # Define the number of models you have
    num_models = len(models_output)
    print(f"Number of models: {num_models}")
    # Create a colormap and generate a list of colors from it
    cmap = cm.get_cmap('Greys', num_models-1)  # You can choose any colormap you like
    colors = cmap(np.linspace(0.3, 1, num_models))

    if plot_type == 'point':
        # Point-to-point comparison (scatter plot)
        plt.figure(figsize=(8, 6))

        for i, model_output in enumerate(models_output):
            color = colors[i] if i < num_models - 1 else 'r'  # Use specified color for all but the last model
            plt.errorbar(model_output, observation_data, yerr=observation_error, fmt='o', label=model_legends[i], capsize=4,
                        ecolor=color, elinewidth=2,mec=color,mfc =color)
        
        # Plot 1:1 line for reference
        # Check if elements in models_output are lists/arrays or just float values
        # Check if models_output is a collection (list/array) or a single float values
        # Flatten models_output elements if they are iterable, otherwise treat them as single values
        flattened_model_output = []
        for m in models_output:
            #print(f"Type of element in models_output: {type(m)}")  # Print type for debugging

            if isinstance(m, (list, np.ndarray)):  # If it's iterable, add elements
                flattened_model_output.extend(m)   # Add the contents of the iterable
            elif isinstance(m, (float, np.float64)):  # If it's a scalar value, append it directly
                flattened_model_output.append(m)
            else:
                raise TypeError(f"Unexpected type in models_output: {type(m)}")  # Raise an error for unexpected types

        # Check if observation_data is iterable, otherwise treat it as a scalar value
        if isinstance(observation_data, (list, np.ndarray)):
            observation_min = min(observation_data)
            observation_max = max(observation_data)
        elif isinstance(observation_data, (float, np.float64)):  # Handle scalar case
            observation_min = observation_data
            observation_max = observation_data
        else:
            raise TypeError(f"Unexpected type for observation_data: {type(observation_data)}")

        # Now calculate min_val using the flattened model output and observation_data
        min_val = min(min(flattened_model_output), observation_min)
        max_val = max(max(flattened_model_output), observation_max)

        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')  # Reference line for perfect agreement
        
        plt.xlabel(model_label)
        plt.ylabel(obs_label)
        plt.title(title)
        # # Adjust the legend: place it outside, split into multiple columns
        # #pdb.set_trace()
        # if num_models > 5:
        #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small', title= None)
        #     # Adjust layout to make room for the legend
        #     plt.tight_layout()
        #     plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for the legend
        # else:
        #     plt.legend()

        #plt.grid(True)
        #plt.show()

    elif plot_type == 'timeseries':
        # Timeseries comparison (line plot)
        if dates is None:
            dates = np.arange(len(observation_data))+start_date  # Default to index if dates are not provided
        
        plt.figure(figsize=(10, 6))
        # Plot each model's output as a line
        #pdb.set_trace()

        if isinstance(models_output, list):
            models_output = np.array(models_output)

        #pdb.set_trace()
        # The number of dates should match the number of data points in the model output
        if models_output.shape[1] != np.size(dates):
            models_output = models_output.T
            # Define the number of models you have
            num_models = len(models_output)
            print(f"Number of models after transpose: {num_models}")
            # Create a colormap and generate a list of colors from it
            cmap = cm.get_cmap('Greys', num_models-1)  # You can choose any colormap you like
            colors = cmap(np.linspace(0.3, 1, num_models))
            #raise ValueError("The number of dates should match the number of data points in the model output.")
        #pdb.set_trace()
        for i, model_output in enumerate(models_output):
            # print(f"Model output shape: {model_output.shape}")
            # print("type of model_output: ", type(model_output))
            # print("type of dates: ", type(dates))
            # print("model_output: ", model_output)
            # print("dates are :",dates)
            linestyle = '-' if i < num_models - 1 else '--'  # Use solid line for all but the last model
            color = colors[i] if i < num_models - 1 else 'r'  # Use specified color for all but the last model
            plt.plot(dates, model_output, label=model_legends[i], linestyle=linestyle, marker=None, color=color)
        
        # Plot the observation data with error bars
            
        plt.errorbar(dates, observation_data, yerr=observation_error, fmt='x', label=obs_label, ecolor='#056eee', elinewidth=2, capsize=4,
                     mec='#056eee',mfc  ='#056eee', alpha=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        #Set the x-ticks and x-tick labels
        plt.xticks(dates, rotation=0)
        # # Adjust the legend: place it outside, split into multiple columns
        # if models_output.shape[0] > 5:
        #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small', title= None)
        #     # Adjust layout to make room for the legend
        #     plt.tight_layout()
        #     plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for the legend
        # else:
        #     plt.legend()
        # plt.grid(True)
        # plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose 'point' for point-to-point comparison or 'timeseries' for timeseries comparison.")

    # Save the figure if save_path is provided
    if save_path and save_name:
        save_name_legend = save_name_legend if save_name_legend else save_name + '_legend.png'
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")


        # Display the plot
        #plt.show()
    # save the legend as a separate file
    if save_path and save_name_legend:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_legend_full = os.path.join(save_path, save_name_legend)
            # Retrieve handles and labels from the current plot
        handles, labels = plt.gca().get_legend_handles_labels()
        # Create a new figure for the legend
        legend_fig = plt.figure(figsize=(10, 6))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis('off')  # Turn off axes for the legend-only figure
        legend = legend_ax.legend(handles, labels, loc='center', frameon=True)
            # Save the legend
        legend_fig.savefig(save_path_legend_full, bbox_inches='tight')
        print(f"Legend saved to {save_path_legend_full}")
        plt.close(legend_fig)  # Close the legend figure



def plot_timeseries_stats(data, start_date=2000, end_date=2019, save_name=None,
                          save_path=None, Y_label=None, F_title='Time Series'):
    """
    Plots a time series of variable with its statistic information (mean/median/std
    /q25/q75/min/max/nmad) with dashed grid lines after each year.

    Parameters:
    - data (numpy array): The variable data.
    - start_date (str) The starting date of the time series in 'YYYY-MM-DD' format.
    - end_date (str) The ending date of the time series in 'YYYY-MM-DD' format.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    - Y_label (str or None): The label for the y-axis. If None, the y-axis label will be 'Variable Name'.
    - F_title (str or None): The title for the figure. If None, the figure title will be 'Time Series'.
    """

    # Create the datetime index with monthly/annual frequency
    date_range = np.arange(start_date,end_date)
    print("date_range is :",date_range)

    # Check if data length matches date range
    if data is None or len(data) != len(date_range):
        raise ValueError("Data length must match the number of time points in the date range.")

    # Create a DataFrame for easier plotting (with years as columns and each row being the data for a year)
    data_df = pd.DataFrame(data.T, columns=date_range)
    # Set up the figure
    plt.figure(figsize=(20, 6))
    # Plot the statistics with dashed grid lines
    
    sns.violinplot(data=data_df, showmeans=False, showmedians=True,fill = False,color = "lightgreen")

    plt.xlabel('Year')

    plt.ylabel(Y_label)

    # Set title
    plt.title(F_title)

    # set the x-axis ticks
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(MultipleLocator(10))  # Major ticks every 10 years
    ax.xaxis.set_minor_locator(MultipleLocator(5))   # Minor ticks every 5 years
    ax.tick_params(axis='x', which='major', length=10, width=1.5)  # Style for major ticks
    ax.tick_params(axis='x', which='minor', length=5, width=1)    # Style for minor ticks

    # add the ygrid
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)

    # Save the figure if save_path is provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")


    # Display the plot
    #plt.show()



def plot_timeseries_stats_sub(data, start_date=2000, end_date=2100, save_name=None,
                          save_path=None, Y_label=None, F_title='Time Series'):
    """
    Plots a time series of variable with its statistic information (mean/median/std
    /q25/q75/min/max/nmad) with subplot for each 20 years.

    Parameters:
    - data (numpy array): The variable data.
    - start_date (str) The starting date of the time series in 'YYYY-MM-DD' format.
    - end_date (str) The ending date of the time series in 'YYYY-MM-DD' format.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    - Y_label (str or None): The label for the y-axis. If None, the y-axis label will be 'Variable Name'.
    - F_title (str or None): The title for the figure. If None, the figure title will be 'Time Series'.
    """

    # Create the datetime index with monthly/annual frequency
    date_range = np.arange(start_date,end_date)
    print("date_range is :",date_range)

    # Check if data length matches date range
    if data is None or len(data) != len(date_range):
        raise ValueError("Data length must match the number of time points in the date range.")

    # Create a DataFrame for easier plotting (with years as columns and each row being the data for a year)
    data_df = pd.DataFrame(data.T, columns=date_range)

    # Convert data into a "long" format DataFrame (similar to the original code with melt)
    data_long = data_df.reset_index().melt(id_vars='index', var_name='Year', value_name='Value')
    data_long.rename(columns={'index': 'Simulation'}, inplace=True)

    # Split the data into 5 chunks (20 years per chunk)
    chunks = [data_long[(data_long['Year'] >= 2000 + i * 20) & (data_long['Year'] < 2000 + (i + 1) * 20)] for i in range(5)]

    # Create subplots (5 rows × 1 column)
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(18, 15), sharex=False)  # Disable sharex
    # Loop through each chunk and plot the violin plot
    for i, chunk in enumerate(chunks):
        # Plot violin plot with 'Year' on x-axis
        sns.violinplot(x='Year', y='Value', data=chunk, color="lightgreen", inner="quartile", ax=axes[i])
        
        # Set titles and labels for each subplot
        axes[i].set_ylabel('Glacier Length Change')
        
        # Set the x-ticks at positions 0, 5, 10, 15, 20 (indices within each chunk)
        axes[i].set_xticks(np.arange(0, 20, 5))  # x-ticks at 0, 5, 10, 15, 20
        
        # Set the x-tick labels to correspond to the correct years
        year_labels = np.arange(2000 + i * 20, 2000 + (i + 1) * 20, 5)  # Correct year labels for each chunk
        axes[i].set_xticklabels(year_labels)  # Set the year labels for the ticks
        
        # Enable minor ticks without labels
        axes[i].tick_params(axis='x', which='minor', length=5, width=1)
        
        # Set major ticks every 5 years on x-axis (0, 5, 10, 15, etc.)
        axes[i].xaxis.set_major_locator(MultipleLocator(5))  # Major ticks at 0, 5, 10, 15, etc.
        axes[i].xaxis.set_minor_locator(MultipleLocator(1))  # Minor ticks every year (for the grid)
        
        # Set x-axis label for the last subplot (shared across all subplots)
        if i == 4:
            axes[i].set_xlabel('Year')
        else:
            axes[i].set_xlabel(None)

    # Adjust the tick labels for the major ticks to display every 5 years correctly
    for ax in axes:
        ax.tick_params(axis='x', which='major', length=10, width=1.5)

    # Add a title for the entire figure
    plt.suptitle(F_title, fontsize=16)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

    # Save the figure if save_path is provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")


    # Display the plot
    #plt.show()



def plot_correlation_matrix(data, save_path=None, save_name=None, title='Correlation Matrix', cmap='coolwarm'):
    """
    Plots a correlation matrix heatmap for the given data.

    Parameters:
    - data (numpy array): The data to plot the correlation matrix for.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.
    - title (str or None): The title for the figure. If None, the figure title will be 'Correlation Matrix'.
    - cmap (str): The colormap to use for the heatmap. Default is 'coolwarm'.

    Returns:
    - None: Displays the plot and optionally saves it.
    """

    # Create a DataFrame for the data
    data_df = pd.DataFrame(data)

    # Calculate the correlation matrix
    corr_matrix = data_df.corr()

    # Set up the figure
    plt.figure(figsize=(10, 8))

    # Plot the correlation matrix as a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)

    # Set the title
    plt.title(title)

    # Save the figure if save_path is provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")

    # Display the plot
    #plt.show()


def plot_relationship_scatters(data, x_label=None, y_label=None, title='Relationship Scatter Plot', save_path=None, save_name=None):
    """
    Plots a scatter plot of the relationship between two variables.

    Parameters:
    - data (numpy array): The data to plot the scatter plot for.
    - x_label (str or None): The label for the x-axis. If None, the x-axis label will be 'X'.
    - y_label (str or None): The label for the y-axis. If None, the y-axis label will be 'Y'.
    - title (str or None): The title for the figure. If None, the figure title will be 'Relationship Scatter Plot'.
    - save_path (str or None): The file path to save the figure. If None, the figure will not be saved.
    - save_name (str or None): The file name to save the figure. If None, the figure will not be saved.

    Returns:
    - None: Displays the plot and optionally saves it.
    """

    # Create a DataFrame for the data
    data_df = pd.DataFrame(data, columns=['X', 'Y'])

    # Set up the figure
    plt.figure(figsize=(8, 6))

    # Plot the scatter plot
    sns.scatterplot(x='X', y='Y', data=data_df)

    # Set the labels and title
    plt.xlabel(x_label if x_label else 'X')
    plt.ylabel(y_label if y_label else 'Y')
    plt.title(title)

    # Save the figure if save_path is provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")

    # Display the plot
    #plt.show()


def plot_relationship_parmeters_result_pairplot(data, parameters, result, save_path=None, save_name=None):
    """
    Plots the relationship between multiple parameters and the result and saves the figure to a specific path.

    Parameters:
    - data (dict or pd.DataFrame): The dataset containing parameters and the result.
    - parameters (list): List of parameter column names to plot against the result.
    - result (str): The result column name.
    - save_path (str): The file path to save the plot.
    - save_name (str): The file name to save the plot.

    Returns:
    - None: Displays the plot and optionally saves it.
    """

    # If the data is a dictionary, convert it to a DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be either a dictionary or a pandas DataFrame.")
    
    # Create a pairplot
    sns.pairplot(df, x_vars=parameters, y_vars=result, kind="scatter")

    # Set the title
    title_name = f"Relationship between Parameters and {result}"

    # Display the plot (optional)
    #plt.show()
    plt.suptitle(title_name, y=1.02)

    if save_name is None:
        save_name = title_name +'_pairplot.png'
    # Save the figure if save_path and save_name are provided
    if save_path and save_name:
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path_full}")


#%% Function to plot the glacier length Timeseries through the model output
def plot_length_TS_Annual (output_path=pygem_prms.main_directory + '/Calibration_AMIS_MB_FA_20002010_N400/',
                           output_fn='calibration_model_Annual_output',rgiid = None, N_iteration = 'Poster',
                           observation_path=pygem_prms.main_directory + '/../lengthchange_data/', save_path=None, save_name=None):
    """
    Plot the glacier length timeseries through the model output

    Parameters:
    - output_path (str): The file path to the model output data.
    - outpuf_fn (str): The file name of the model output data.
    - rgiid (str): The RGI ID of the glacier.
    - N_iteration (str): The number of iteration for the model output. The default is 'Poster', for the lastest iteration.
    - observation_path (str): The file path to the observation data.
    - save_path (str): The file path to save the plot.
    - save_name (str): The file name to save the plot.
    """
    
    # Load the model output, the json file
    output_filename = f"modeloutput/Annual/{output_fn}_{rgiid}_{N_iteration}.json" 
    output_fp_annual = os.path.join(output_path, output_filename)

    try:
        with open(output_fp_annual, 'r', encoding='utf-8') as f:
            output_data_annual = json.load(f)
    except:
        print(f"Error loading JSON file: {output_fp_annual}")
        output_data_annual = None  # or an empty dict {} if needed



    # ==== restruct based on the weights (read the Unique info of Model posterior parameters)
    output_folder_post_params_unique = os.path.join(output_path, 'parameter','Poster','Unique')
    output_filename_params_unique_Info = f'calibration_poster_Params_unique_{rgiid}_Info.json'
    output_fp_params_unique_Info = os.path.join(output_folder_post_params_unique, output_filename_params_unique_Info) 
    
    with open(output_fp_params_unique_Info, 'r') as f:
        parms_UniqInfo_dict = json.load(f)
    unique_counts= parms_UniqInfo_dict['unique_counts']
    #pdb.set_trace()
    # read the model length change data
    lengthchange_dLdt_model_array_annual = np.array(output_data_annual['lengthchange_dLdt_model_array_annual_myr'])
    lengthchange_m_TMS_model_array_annual = np.array(output_data_annual['lengthchange_m_TMS_model_array_annual'])
    # repeat the model output based on the unique counts
    lengthchange_dLdt_model_array_annual_post = np.repeat(lengthchange_dLdt_model_array_annual,unique_counts,axis = 1)
    lengthchange_m_TMS_model_array_annual_post = np.repeat(lengthchange_m_TMS_model_array_annual,unique_counts,axis = 1)



    # Load the observation data #TODO at the moment, read the observation data from the csv file, should be a uniform for the regional data
    observation_annual_fn = 'lengthchange_annual_'+pygem_prms.glac_no[0].split('.')[0]+'_'+ pygem_prms.glac_no[0].split('.')[1]+'.csv'
    observation_annual_fp = os.path.join(observation_path, observation_annual_fn)
    observation_data = pd.read_csv(observation_annual_fp)
    for x in observation_data.RGIId.values:
        if x == rgiid:
            observation_data_annual = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr']
            observation_data_annual_unc = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr_unc']
            break
        else:
            observation_data_annual = None
            observation_data_annual_unc = None
            print(f"Error: No observation data found for RGI ID {rgiid}")

    # Convert each string representation of a list into an actual Python list
    observation_data_annual_list = observation_data_annual.apply(ast.literal_eval)
    observation_data_annual_unc_list = observation_data_annual_unc.apply(ast.literal_eval)
    # Convert list-like Series elements into a NumPy array
    observation_data_annual_array = np.array(observation_data_annual_list.tolist(), dtype=float)
    observation_data_annual_unc_array = np.array(observation_data_annual_unc_list.tolist(), dtype=float)
    #%% ==== Plot the model output and observation data

    # ===== the cumulative lengthchange of the model output and observation data #TODO: check the dimention of the data
    delt_lengthchange_dLdt_model_array_annual_post = np.cumsum(lengthchange_dLdt_model_array_annual_post,axis=0)
    delt_lengthchange_m_TMS_model_array_annual_post = np.cumsum(lengthchange_m_TMS_model_array_annual_post,axis=0)
    delt_lengthchange_m_observation_annual = np.cumsum(observation_data_annual_array)
    delt_lengthchange_m_observation_unc_annual =  np.sqrt(np.cumsum(observation_data_annual_unc_array**2))  # Propagating uncertainty


    # ===== the length varation of the model output and observation data
    # convert the lengthchange to length
    length_original = 53600
    length_dLdt_annual_m = length_original + delt_lengthchange_dLdt_model_array_annual_post
    length_TMS_annual_m = length_original + delt_lengthchange_m_TMS_model_array_annual_post
    # convert the observation data to length
    length_observation_annual = length_original + delt_lengthchange_m_observation_annual
    length_observation_unc_annual = delt_lengthchange_m_observation_unc_annual
    length_observation_annual = length_observation_annual.flatten()
    length_observation_unc_annual = length_observation_unc_annual.flatten()
    delt_lengthchange_m_observation_annual = delt_lengthchange_m_observation_annual.flatten()
    delt_lengthchange_m_observation_unc_annual = delt_lengthchange_m_observation_unc_annual.flatten()
    #pdb.set_trace()
    # ===== plot the cumulative lengthchanges of the model output and observation data
    
    # TODO the date should be dynamic, and read from the model output
    X_Years = np.arange(2000, 2000 + length_observation_annual.shape[0])

    # ==== plot the length based on dLdt with the obersevation data in subplot 1, and the length based on TMS with the obersevation data in subplot 2
    # subplot setting



    # ===== plot the cumulative length varation of the model output and observation data
    #pdb.set_trace()
    # Create a figure with two vertically stacked subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), sharex=True)
    # First subplot
    axes[0,0].plot(X_Years, length_dLdt_annual_m, label='Model dLdt')
    axes[0,0].plot(X_Years, length_observation_annual, label='Observation')
    axes[0,0].fill_between(X_Years, length_observation_annual  - length_observation_unc_annual ,
                         length_observation_annual  + length_observation_unc_annual , color='gray', alpha=0.5)
    axes[0,0].set_ylabel('Length (m)')
    #axes[0,0].legend()
    axes[0,0].grid(False)
    #axes[0,0].set_title('Model & Observation Comparison of glacier length (centerline) based on dLdt')
    # Second subplot
    axes[0,1].plot(X_Years, length_TMS_annual_m, label='Modeled flowline')
    axes[0,1].plot(X_Years, length_observation_annual, label='Observation')
    axes[0,1].fill_between(X_Years, length_observation_annual - length_observation_unc_annual,
                         length_observation_annual + length_observation_unc_annual, color='gray', alpha=0.5)
    #axes[0,1].set_xlabel('Year')
    #axes[0,1].set_ylabel('Length (m)')
    #axes[0,1].legend()
    axes[0,1].grid(False)
    #axes[0,1].set_title('Model & Observation Comparison of glacier length (centerline) based on flowline model')

    # Third subplot
    axes[1,0].plot(X_Years, delt_lengthchange_dLdt_model_array_annual_post, label='Model dLdt')
    axes[1,0].plot(X_Years, delt_lengthchange_m_observation_annual, label='Observation')
    axes[1,0].fill_between(X_Years, delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                         delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, color='gray', alpha=0.5)
    axes[1,0].set_ylabel('Length change (m)')
    axes[1,0].set_xlabel('Year')
    #axes[1,0].legend()
    axes[1,0].grid(False)
    #axes[1,0].set_title('Model & Observation Comparison of glacier length change (centerline) based on dLdt')
    # Fourth subplot
    axes[1,1].plot(X_Years, delt_lengthchange_m_TMS_model_array_annual_post, label='Modeled flowline')
    axes[1,1].plot(X_Years, delt_lengthchange_m_observation_annual, label='Observation')
    axes[1,1].fill_between(X_Years, delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                         delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, color='gray', alpha=0.5)
    axes[1,1].set_xlabel('Year')
    #axes[1,1].set_ylabel('Length change (m)')
    #axes[1,1].legend()
    axes[1,1].grid(False)
    #axes[1,1].set_title('Model & Observation Comparison of glacier length change (centerline) based on flowline

    # Set x-axis limits and ticks for all subplots
    for ax in axes.flat:
        ax.set_xlim(2000, 2020)
        ax.set_xticks(X_Years)

    # Adjust layout properly
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Add padding to prevent overlap
    # Save the figure if save_path is provided
    if save_path == None:
        save_path = output_path + '/figures/'
    if save_name == None:
        save_name = 'Glacier_length_lengthchange_timeseries'


    if save_path and save_name:
        save_name_length= save_name + '.png'
        save_path_full = os.path.join(save_path, save_name_length)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")
    # Display the plot
    #plt.show()





def plot_length_TS_Annual_New(output_path=pygem_prms.main_directory + '/Calibration_AMIS_MB_FA_20002010_N400/',
                           output_fn='calibration_model_Annual_output',rgiid = None, N_iteration = 'Poster',
                           observation_path=pygem_prms.main_directory + '/../lengthchange_data/', save_path=None, save_name=None):
    """
    Plot the cumulative glacier length timeseries from model output and observation data.

    Parameters:
    - output_path (str): Path to model output data.
    - output_fn (str): File name of the model output data.
    - rgiid (str): Glacier ID.
    - N_iteration (str): Iteration number for model output (default='Poster').
    - observation_path (str): Path to observation data.
    - save_path (str): Path to save the plot.
    - save_name (str): File name to save the plot.
    """

    # Load model output from JSON file
    output_filename = f"modeloutput/Annual/{output_fn}_{rgiid}_{N_iteration}.json" 
    output_fp_annual = os.path.join(output_path, output_filename)

    try:
        with open(output_fp_annual, 'r', encoding='utf-8') as f:
            output_data_annual = json.load(f)
    except:
        print(f"Error loading JSON file: {output_fp_annual}")
        output_data_annual = None  # or an empty dict {} if needed

    # Load model length change data
    lengthchange_dLdt_model_array_annual = np.array(output_data_annual['lengthchange_dLdt_model_array_annual_myr'])
    lengthchange_m_TMS_model_array_annual = np.array(output_data_annual['lengthchange_m_TMS_model_array_annual'])

    # Load observation data
    # Load the observation data #TODO at the moment, read the observation data from the csv file, should be a uniform for the regional data
    observation_annual_fn = 'lengthchange_annual_'+pygem_prms.glac_no[0].split('.')[0]+'_'+ pygem_prms.glac_no[0].split('.')[1]+'.csv'
    observation_annual_fp = os.path.join(observation_path, observation_annual_fn)
    observation_data = pd.read_csv(observation_annual_fp)
    for x in observation_data.RGIId.values:
        if x == rgiid:
            observation_data_annual = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr']
            observation_data_annual_unc = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr_unc']
            break
        else:
            observation_data_annual = None
            observation_data_annual_unc = None
            print(f"Error: No observation data found for RGI ID {rgiid}")

    # Convert observation data from string to numerical lists
    observation_data_annual = np.array(observation_data_annual.apply(ast.literal_eval).tolist(), dtype=float)
    observation_data_annual_unc = np.array(observation_data_annual_unc.apply(ast.literal_eval).tolist(), dtype=float)

    # Compute cumulative sum for model and observation
    delt_lengthchange_dLdt_model_array_annual = np.cumsum(lengthchange_dLdt_model_array_annual, axis=0)
    delt_lengthchange_m_TMS_model_array_annual = np.cumsum(lengthchange_m_TMS_model_array_annual, axis=0)
    delt_lengthchange_m_observation_annual = np.cumsum(observation_data_annual)
    delt_lengthchange_m_observation_unc_annual = np.sqrt(np.cumsum(observation_data_annual_unc**2))

    # Define years dynamically
    X_Years = np.arange(2000, 2000 + delt_lengthchange_dLdt_model_array_annual.shape[0])

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))

    # Model output as grey ensemble lines
    ax.plot(X_Years, delt_lengthchange_dLdt_model_array_annual, color='grey', alpha=0.3, label='_nolegend_')

    # Observation data in blue with uncertainty shading
    ax.plot(X_Years, delt_lengthchange_m_observation_annual, color='blue', label='Observation')
    ax.fill_between(X_Years,
                    delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                    delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual,
                    color='blue', alpha=0.3)

    # Labels and legend
    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Length Change (m)")
    ax.set_title(f"Glacier Length Change for {rgiid}")
    ax.legend()
    ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{save_name or 'glacier_length_timeseries'}.png")
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Figure saved to {save_file}")
    #
    plt.show()
 


import os
import json
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt



def plot_length_TS_Annual_New2(output_path=pygem_prms.main_directory + '/Calibration_AMIS_MB_FA_20002010_N400/',
                           output_fn='calibration_model_Annual_output',rgiid = None, N_iteration = 'Poster',
                           observation_path=pygem_prms.main_directory + '/../lengthchange_data/', save_path=None, save_name=None):
    """
    Plot the cumulative glacier length timeseries from model output and observation data in 4 subplots.

    Parameters:
    - output_path (str): Path to model output data.
    - output_fn (str): File name of the model output data.
    - rgiid (str): Glacier ID.
    - N_iteration (str): Iteration number for model output.
    - observation_path (str): Path to observation data.
    - save_path (str): Path to save the plot.
    - save_name (str): File name to save the plot.
    """

    if not rgiid:
        print("Error: RGI ID must be provided.")
        return

    # Load model output from JSON file
    output_filename = f"modeloutput/Annual/{output_fn}_{rgiid}_{N_iteration}.json" 
    output_fp_annual = os.path.join(output_path, output_filename)

    try:
        with open(output_fp_annual, 'r', encoding='utf-8') as f:
            output_data_annual = json.load(f)
    except:
        print(f"Error loading JSON file: {output_fp_annual}")
        output_data_annual = None  # or an empty dict {} if needed

    # Load model length change data
    lengthchange_dLdt_model_array_annual = np.array(output_data_annual['lengthchange_dLdt_model_array_annual_myr'])
    lengthchange_m_TMS_model_array_annual = np.array(output_data_annual['lengthchange_m_TMS_model_array_annual'])

    # Load observation data
    # Load the observation data #TODO at the moment, read the observation data from the csv file, should be a uniform for the regional data
    observation_annual_fn = 'lengthchange_annual_'+pygem_prms.glac_no[0].split('.')[0]+'_'+ pygem_prms.glac_no[0].split('.')[1]+'.csv'
    observation_annual_fp = os.path.join(observation_path, observation_annual_fn)
    observation_data = pd.read_csv(observation_annual_fp)
    for x in observation_data.RGIId.values:
        if x == rgiid:
            observation_data_annual = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr']
            observation_data_annual_unc = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr_unc']
            break
        else:
            observation_data_annual = None
            observation_data_annual_unc = None
            print(f"Error: No observation data found for RGI ID {rgiid}")

    # Convert observation data from string to numerical lists
    observation_data_annual = np.array(observation_data_annual.apply(ast.literal_eval).tolist(), dtype=float)
    observation_data_annual_unc = np.array(observation_data_annual_unc.apply(ast.literal_eval).tolist(), dtype=float)

    # Compute cumulative sum for model and observation
    delt_lengthchange_dLdt_model_array_annual = np.cumsum(lengthchange_dLdt_model_array_annual, axis=0)
    delt_lengthchange_m_TMS_model_array_annual = np.cumsum(lengthchange_m_TMS_model_array_annual, axis=0)
    delt_lengthchange_m_observation_annual = np.cumsum(observation_data_annual)
    delt_lengthchange_m_observation_unc_annual = np.sqrt(np.cumsum(observation_data_annual_unc**2))

    # Convert length change to absolute glacier length
    initial_length = 53600  # Initial glacier length (adjust if needed)
    length_dLdt_annual_m = initial_length + delt_lengthchange_dLdt_model_array_annual
    length_TMS_annual_m = initial_length + delt_lengthchange_m_TMS_model_array_annual
    length_observation_annual = initial_length + delt_lengthchange_m_observation_annual

    # Define years dynamically
    X_Years = np.arange(2000, 2000 + delt_lengthchange_dLdt_model_array_annual.shape[0])

    # Plot setup (4 subplots)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)

    # Function to set x-axis labels and ticks
    for ax in axes.flat:
        ax.set_xticks(X_Years)  # Set ticks annually
        ax.set_xticklabels([str(year) if year % 2 == 0 else "" for year in X_Years])

    # First subplot: Model (dLdt) vs Observation - Absolute Length
    axes[0,0].plot(X_Years, length_dLdt_annual_m, color='grey', alpha=0.5)
    axes[0,0].plot(X_Years, length_observation_annual, color='blue', label='Observed')
    axes[0,0].fill_between(X_Years, 
                           length_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           length_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    axes[0,0].set_ylabel("Glacier Length (m)")
    # custom legend
    ensemble_legend = mlines.Line2D([], [], color='grey', alpha=0.8, label='Modeled')  # Proxy for ensemble
    obs_legend = mlines.Line2D([], [], color='blue', label='Observed')  # Proxy for observation
    axes[0,0].legend(handles=[ensemble_legend, obs_legend], loc='best')  # Include both in legend
    # ensemble_legend = mlines.Line2D([], [], color='grey', alpha=0.8, label='Modeled')  # Proxy artist
    # axes[0,0].legend(handles=[ensemble_legend], loc='best')  # Add custom legend
    #axes[0,0].set_title("Model (dLdt) vs Observation - Length")

    # Second subplot: Model (TMS) vs Observation - Absolute Length
    axes[0,1].plot(X_Years, length_TMS_annual_m, color='grey', alpha=0.5)
    axes[0,1].plot(X_Years, length_observation_annual, color='blue', label='Observed')
    axes[0,1].fill_between(X_Years, 
                           length_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           length_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    #axes[0,1].set_title("Model (TMS) vs Observation - Length")

    # Third subplot: Model (dLdt) vs Observation - Cumulative Length Change
    axes[1,0].plot(X_Years, delt_lengthchange_dLdt_model_array_annual, color='grey', alpha=0.5)
    axes[1,0].plot(X_Years, delt_lengthchange_m_observation_annual, color='blue', label='Observed')
    axes[1,0].fill_between(X_Years, 
                           delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    axes[1,0].set_xlabel("Year")
    axes[1,0].set_ylabel("Cumulative Length Change (m)")
    #axes[1,0].legend()
    #axes[1,0].set_title("Model (dLdt) vs Observation - Change")

    # Fourth subplot: Model (TMS) vs Observation - Cumulative Length Change
    axes[1,1].plot(X_Years, delt_lengthchange_m_TMS_model_array_annual, color='grey', alpha=0.5)
    axes[1,1].plot(X_Years, delt_lengthchange_m_observation_annual, color='blue', label='Observed')
    axes[1,1].fill_between(X_Years, 
                           delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    axes[1,1].set_xlabel("Year")
    #axes[1,1].set_title("Model (TMS) vs Observation - Change")

    # Adjust layout
    plt.tight_layout()

    # Save figure if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{save_name or 'glacier_length_timeseries_New2'}.png")
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Figure saved to {save_file}")

    plt.show()





def plot_length_TS_Annual_New3(output_path= '/Calibration_AMIS_MB_FA_20002010_N200/',
                           output_fn='calibration_model_Annual_output',rgiid = None, N_iteration = 'Poster',
                           observation_path=pygem_prms.main_directory + '/../lengthchange_data/', save_path=None, save_name=None):
    """
    Plot the cumulative glacier length timeseries from model output and observation data in 4 subplots.

    Parameters:
    - output_path (str): Path to model output data.
    - output_fn (str): File name of the model output data.
    - rgiid (str): Glacier ID.
    - N_iteration (str): Iteration number for model output.
    - observation_path (str): Path to observation data.
    - save_path (str): Path to save the plot.
    - save_name (str): File name to save the plot.
    """

    if not rgiid:
        print("Error: RGI ID must be provided.")
        return

    output_path = pygem_prms.main_directory + output_path
    # Load model output from JSON file
    output_filename = f"modeloutput/Annual/{output_fn}_{rgiid}_{N_iteration}.json" 
    output_fp_annual = os.path.join(output_path, output_filename)

    try:
        with open(output_fp_annual, 'r', encoding='utf-8') as f:
            output_data_annual = json.load(f)
    except:
        print(f"Error loading JSON file: {output_fp_annual}")
        output_data_annual = None  # or an empty dict {} if needed

    # Load model length change data
    lengthchange_dLdt_model_array_annual = np.array(output_data_annual['lengthchange_dLdt_model_array_annual_myr'])
    lengthchange_m_TMS_model_array_annual = np.array(output_data_annual['lengthchange_m_TMS_model_array_annual'])

    # ==== restruct based on the weights (read the Unique info of Model posterior parameters)
    output_folder_post_params_unique = os.path.join(output_path, 'parameter','Poster','Unique')
    output_filename_params_unique_Info = f'calibration_poster_Params_unique_{rgiid}_Info.json'
    output_fp_params_unique_Info = os.path.join(output_folder_post_params_unique, output_filename_params_unique_Info) 
    
    with open(output_fp_params_unique_Info, 'r') as f:
        parms_UniqInfo_dict = json.load(f)
    unique_counts= parms_UniqInfo_dict['unique_counts']

    # repeat the model output based on the unique counts
    lengthchange_dLdt_model_array_annual_post = np.repeat(lengthchange_dLdt_model_array_annual,unique_counts,axis = 1)
    lengthchange_m_TMS_model_array_annual_post = np.repeat(lengthchange_m_TMS_model_array_annual,unique_counts,axis = 1)


    # Load observation data
    # Load the observation data #TODO at the moment, read the observation data from the csv file, should be a uniform for the regional data
    observation_annual_fn = 'lengthchange_annual_'+pygem_prms.glac_no[0].split('.')[0]+'_'+ pygem_prms.glac_no[0].split('.')[1]+'.csv'
    observation_annual_fp = os.path.join(observation_path, observation_annual_fn)
    observation_data = pd.read_csv(observation_annual_fp)
    for x in observation_data.RGIId.values:
        if x == rgiid:
            observation_data_annual = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr']
            observation_data_annual_unc = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr_unc']
            break
        else:
            observation_data_annual = None
            observation_data_annual_unc = None
            print(f"Error: No observation data found for RGI ID {rgiid}")

    # Convert observation data from string to numerical lists
    observation_data_annual = np.array(observation_data_annual.apply(ast.literal_eval).tolist(), dtype=float)
    observation_data_annual_unc = np.array(observation_data_annual_unc.apply(ast.literal_eval).tolist(), dtype=float)

    # Compute cumulative sum for model and observation
    delt_lengthchange_dLdt_model_array_annual = np.cumsum(lengthchange_dLdt_model_array_annual_post, axis=0)
    delt_lengthchange_m_TMS_model_array_annual = np.cumsum(lengthchange_m_TMS_model_array_annual_post, axis=0)
    delt_lengthchange_m_observation_annual = np.cumsum(observation_data_annual)
    delt_lengthchange_m_observation_unc_annual = np.sqrt(np.cumsum(observation_data_annual_unc**2))

    # Convert length change to absolute glacier length
    initial_length = 53600  # Initial glacier length (adjust if needed)
    length_dLdt_annual_m = initial_length + delt_lengthchange_dLdt_model_array_annual
    length_TMS_annual_m = initial_length + delt_lengthchange_m_TMS_model_array_annual
    length_observation_annual = initial_length + delt_lengthchange_m_observation_annual

    # add the initial value, extend the length to the initial year
    length_dLdt_annual_m = np.vstack([np.full((1, length_dLdt_annual_m.shape[1]), initial_length), length_dLdt_annual_m])
    length_TMS_annual_m = np.vstack([np.full((1, length_TMS_annual_m.shape[1]), initial_length), length_TMS_annual_m])
    length_observation_annual = np.hstack([initial_length, length_observation_annual])

    # add the initial delt value as 0 for the initial year
    delt_lengthchange_dLdt_model_array_annual = np.vstack([np.zeros((1, delt_lengthchange_dLdt_model_array_annual.shape[1])), delt_lengthchange_dLdt_model_array_annual])
    delt_lengthchange_m_TMS_model_array_annual = np.vstack([np.zeros((1, delt_lengthchange_m_TMS_model_array_annual.shape[1])), delt_lengthchange_m_TMS_model_array_annual])
    delt_lengthchange_m_observation_annual = np.hstack([0, delt_lengthchange_m_observation_annual])

    # add the initial uncertainty as same as the first year for the initial year
    delt_lengthchange_m_observation_unc_annual = np.hstack([observation_data_annual_unc[0,0], delt_lengthchange_m_observation_unc_annual])    
    #pdb.set_trace()

    # Define years dynamically
    X_Years = np.arange(2000, 2000 + delt_lengthchange_dLdt_model_array_annual.shape[0])

    # Plot setup (4 subplots)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex=True)

    # Function to set x-axis labels and ticks
    for ax in axes.flat:
        ax.set_xticks(X_Years)  # Set ticks annually
        ax.set_xticklabels([str(year) if year % 2 == 0 else "" for year in X_Years])

    # First subplot: Model (dLdt) vs Observation - Absolute Length
    axes[0,0].plot(X_Years, length_dLdt_annual_m, color='grey', alpha=0.5)
    axes[0,0].plot(X_Years, length_observation_annual, color='blue', label='Observed')
    axes[0,0].fill_between(X_Years, 
                           length_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           length_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    axes[0,0].set_ylabel("Glacier Length (m)")
    # custom legend
    ensemble_legend = mlines.Line2D([], [], color='grey', alpha=0.8, label='Modeled')  # Proxy for ensemble
    obs_legend = mlines.Line2D([], [], color='blue', label='Observed')  # Proxy for observation
    axes[0,0].legend(handles=[ensemble_legend, obs_legend], loc='best')  # Include both in legend
    # ensemble_legend = mlines.Line2D([], [], color='grey', alpha=0.8, label='Modeled')  # Proxy artist
    # axes[0,0].legend(handles=[ensemble_legend], loc='best')  # Add custom legend
    #axes[0,0].set_title("Model (dLdt) vs Observation - Length")

    # Second subplot: Model (TMS) vs Observation - Absolute Length
    axes[0,1].plot(X_Years, length_TMS_annual_m, color='grey', alpha=0.5)
    axes[0,1].plot(X_Years, length_observation_annual, color='blue', label='Observed')
    axes[0,1].fill_between(X_Years, 
                           length_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           length_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    #axes[0,1].set_title("Model (TMS) vs Observation - Length")

    # Third subplot: Model (dLdt) vs Observation - Cumulative Length Change
    axes[1,0].plot(X_Years, delt_lengthchange_dLdt_model_array_annual, color='grey', alpha=0.5)
    axes[1,0].plot(X_Years, delt_lengthchange_m_observation_annual, color='blue', label='Observed')
    axes[1,0].fill_between(X_Years, 
                           delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    axes[1,0].set_xlabel("Year")
    axes[1,0].set_ylabel("Cumulative Length Change (m)")
    #axes[1,0].legend()
    #axes[1,0].set_title("Model (dLdt) vs Observation - Change")

    # Fourth subplot: Model (TMS) vs Observation - Cumulative Length Change
    axes[1,1].plot(X_Years, delt_lengthchange_m_TMS_model_array_annual, color='grey', alpha=0.5)
    axes[1,1].plot(X_Years, delt_lengthchange_m_observation_annual, color='blue', label='Observed')
    axes[1,1].fill_between(X_Years, 
                           delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='blue', alpha=0.3)
    axes[1,1].set_xlabel("Year")
    #axes[1,1].set_title("Model (TMS) vs Observation - Change")

    # **Set y-axis limits to be the same for upper and lower panels**
    # Upper panels (absolute length)
    upper_min = min(axes[0,0].get_ylim()[0], axes[0,1].get_ylim()[0])
    upper_max = max(axes[0,0].get_ylim()[1], axes[0,1].get_ylim()[1])
    axes[0,0].set_ylim(upper_min, upper_max)
    axes[0,1].set_ylim(upper_min, upper_max)

    # Bottom panels (cumulative length change)
    bottom_min = min(axes[1,0].get_ylim()[0], axes[1,1].get_ylim()[0])
    bottom_max = max(axes[1,0].get_ylim()[1], axes[1,1].get_ylim()[1])
    axes[1,0].set_ylim(bottom_min, bottom_max)
    axes[1,1].set_ylim(bottom_min, bottom_max)
    # Adjust layout
    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path == None:
        save_path = output_path + '/figures/'
    if save_name == None:
        save_name = 'Glacier_length_lengthchange_timeseries'
    # Save figure if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{save_name or 'glacier_length_timeseries'}.png")
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Figure saved to {save_file}")

    #plt.show()



def plot_length_dl_TS_Annual(output_path= '/Calibration_AMIS_MB_FA_20002010_N200/',
                           output_fn='calibration_model_Annual_output',rgiid = None, N_iteration = 'Poster',
                           observation_path=pygem_prms.main_directory + '/../lengthchange_data/', save_path=None, save_name=None):
    """
    Plot the cumulative glacier length/dl timeseries from model output and observation data in 3 subplots.

    Parameters:
    - output_path (str): Path to model output data.
    - output_fn (str): File name of the model output data.
    - rgiid (str): Glacier ID.
    - N_iteration (str): Iteration number for model output.
    - observation_path (str): Path to observation data.
    - save_path (str): Path to save the plot.
    - save_name (str): File name to save the plot.
    """

    if not rgiid:
        print("Error: RGI ID must be provided.")
        return

    output_path = pygem_prms.main_directory + output_path
    # Load model output from JSON file
    output_filename = f"modeloutput/Annual/{output_fn}_{rgiid}_{N_iteration}.json" 
    output_fp_annual = os.path.join(output_path, output_filename)

    try:
        with open(output_fp_annual, 'r', encoding='utf-8') as f:
            output_data_annual = json.load(f)
    except:
        print(f"Error loading JSON file: {output_fp_annual}")
        output_data_annual = None  # or an empty dict {} if needed

    # Load model length change data
    lengthchange_dLdt_model_array_annual = np.array(output_data_annual['lengthchange_dLdt_model_array_annual_myr'])
    lengthchange_m_TMS_model_array_annual = np.array(output_data_annual['lengthchange_m_TMS_model_array_annual'])

    # ==== restruct based on the weights (read the Unique info of Model posterior parameters)
    output_folder_post_params_unique = os.path.join(output_path, 'parameter','Poster','Unique')
    output_filename_params_unique_Info = f'calibration_poster_Params_unique_{rgiid}_Info.json'
    output_fp_params_unique_Info = os.path.join(output_folder_post_params_unique, output_filename_params_unique_Info) 
    
    with open(output_fp_params_unique_Info, 'r') as f:
        parms_UniqInfo_dict = json.load(f)
    unique_counts= parms_UniqInfo_dict['unique_counts']

    # repeat the model output based on the unique counts
    lengthchange_dLdt_model_array_annual_post = np.repeat(lengthchange_dLdt_model_array_annual,unique_counts,axis = 1)
    lengthchange_m_TMS_model_array_annual_post = np.repeat(lengthchange_m_TMS_model_array_annual,unique_counts,axis = 1)


    # Load observation data
    # Load the observation data #TODO at the moment, read the observation data from the csv file, should be a uniform for the regional data
    observation_annual_fn = 'lengthchange_annual_'+pygem_prms.glac_no[0].split('.')[0]+'_'+ pygem_prms.glac_no[0].split('.')[1]+'.csv'
    observation_annual_fp = os.path.join(observation_path, observation_annual_fn)
    observation_data = pd.read_csv(observation_annual_fp)
    for x in observation_data.RGIId.values:
        if x == rgiid:
            observation_data_annual = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr']
            observation_data_annual_unc = observation_data[observation_data.RGIId == x]['dLdt_m_per_yr_unc']
            break
        else:
            observation_data_annual = None
            observation_data_annual_unc = None
            print(f"Error: No observation data found for RGI ID {rgiid}")

    # Convert observation data from string to numerical lists
    observation_data_annual = np.array(observation_data_annual.apply(ast.literal_eval).tolist(), dtype=float)
    observation_data_annual_unc = np.array(observation_data_annual_unc.apply(ast.literal_eval).tolist(), dtype=float)

    # Compute cumulative sum for model and observation
    delt_lengthchange_dLdt_model_array_annual = np.cumsum(lengthchange_dLdt_model_array_annual_post, axis=0)
    delt_lengthchange_m_TMS_model_array_annual = np.cumsum(lengthchange_m_TMS_model_array_annual_post, axis=0)
    delt_lengthchange_m_observation_annual = np.cumsum(observation_data_annual)
    delt_lengthchange_m_observation_unc_annual = np.sqrt(np.cumsum(observation_data_annual_unc**2))

    # Convert length change to absolute glacier length
    initial_length = 53600  # Initial glacier length (adjust if needed)
    length_dLdt_annual_m = initial_length + delt_lengthchange_dLdt_model_array_annual
    length_TMS_annual_m = initial_length + delt_lengthchange_m_TMS_model_array_annual
    length_observation_annual = initial_length + delt_lengthchange_m_observation_annual

    #pdb.set_trace()
    # Define years dynamically
    X_Years = np.arange(2000, 2000 + delt_lengthchange_dLdt_model_array_annual.shape[0])

    # Plot setup (4 subplots)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=True)

    # Function to set x-axis labels and ticks
    for ax in axes.flat:
        ax.set_xticks(X_Years)  # Set ticks annually
        ax.set_xticklabels([str(year) if year % 2 == 0 else "" for year in X_Years])

    # First subplot: Model (dLdt) vs Observation - Length change
    #pdb.set_trace()
    axes[0].plot(X_Years, lengthchange_dLdt_model_array_annual_post, color='grey', alpha=0.5)
    axes[0].errorbar(X_Years, observation_data_annual.squeeze(), yerr=observation_data_annual_unc.squeeze(), fmt='x',
                       label='Observed', ecolor='#056eee', elinewidth=2, capsize=4,mec='#056eee',mfc  ='#056eee', alpha=1)

    
    axes[0].set_ylabel("Length change rate (m a⁻¹)")
    # custom legend
    ensemble_legend = mlines.Line2D([], [], color='grey', alpha=0.8, label='Modeled')  # Proxy for ensemble
    obs_legend = mlines.Line2D([], [], color='#056eee', label='Observed')  # Proxy for observation
    axes[0].legend(handles=[ensemble_legend, obs_legend], loc='best', fontsize=16)  # Include both in legend

    # Second subplot: Model (dLdt) vs Observation - Cumulative Length Change
    axes[1].plot(X_Years, delt_lengthchange_dLdt_model_array_annual, color='grey', alpha=0.5)
    axes[1].plot(X_Years, delt_lengthchange_m_observation_annual, color='#056eee', label='Observed')
    axes[1].fill_between(X_Years, 
                           delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='#056eee', alpha=0.3)
    axes[1].set_ylabel("Cumulative length Change (m)")


    # Third subplot: Model (TMS) vs Observation - Cumulative Length Change
    axes[2].plot(X_Years, delt_lengthchange_m_TMS_model_array_annual, color='grey', alpha=0.5)
    axes[2].plot(X_Years, delt_lengthchange_m_observation_annual, color='#056eee', label='Observed')
    axes[2].fill_between(X_Years, 
                           delt_lengthchange_m_observation_annual - delt_lengthchange_m_observation_unc_annual,
                           delt_lengthchange_m_observation_annual + delt_lengthchange_m_observation_unc_annual, 
                           color='#056eee', alpha=0.3)
    axes[2].set_ylabel("Cumulative length Change (m)")
    axes[2].set_xlabel("Year")


    # **Set y-axis limits to be the same for upper and lower panels**
    # Bottom panels (cumulative length change)
    bottom_min = min(axes[1].get_ylim()[0], axes[2].get_ylim()[0])
    bottom_max = max(axes[1].get_ylim()[1], axes[2].get_ylim()[1])
    axes[1].set_ylim(bottom_min, bottom_max)
    axes[2].set_ylim(bottom_min, bottom_max)

    # Set the Axis properties
    # Set font size and font family for x-axis and y-axis labels
    for ax in axes:
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
    # Add "SERMeQ" in the bottom-left of axes[1]
    axes[1].text(
        0.02, 0.05, "SERMeQ",  
        transform=axes[1].transAxes,  
        fontsize=16,  
        color='black'
    )

    # Add "Flowline model profile" in the bottom-left of axes[2]
    axes[2].text(
        0.02, 0.05, "Flowline model profile",  
        transform=axes[2].transAxes,  
        fontsize=16,  
        color='black'
    )

    # Adjust layout
    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path == None:
        save_path = output_path + '/figures/'
    if save_name == None:
        save_name = 'Glacier_length_lengthchange_timeseries'
    # Save figure if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, save_name + '.png')
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Figure saved to {save_file}")


# function to plot the CDF and one-to-one comparison of observation and modeled value
def plot_cdf_and_one_to_one(observed_df = None, modeled_df = None, modeled_df_raw= None,
                            obs_name = None, obs_unc_name = None, modeled_key = None, item_name = None,
                            period = None, Xlabel = None, Ylabel = None,Xlim = None, Ylim = None,
                            subplot_label_L ='a', subplot_label_R = 'b',title= None, save_path=None, save_name=None,
                            legend_index = True):
    """
    Plots the CDF for observed and modeled values,
    and a one-to-one comparison with box plots for modeled results and error bars for observations.
    Parameters:
    - observed_df (pd.DataFrame): DataFrame containing observed values with columns e.g. 'rgiid', 'length_change', 'length_change_unc'.
    - modeled_df (pd.DataFrame): DataFrame containing modeled values with columns 'rgiid', 'mean', 'hdi_95_low', 'hdi_95_high'.
    - modeled_df_raw (pd.DataFrame): DataFrame containing raw modeled values for box plots, with 'rgiid' and 'lengthchange_dLdt_model_array_annual_myr' columns.
    - obs_name (str): Name of the observed length change column in observed_df, e.g. 'length_change'.
    - obs_unc_name (str): Name of the observed length change uncertainty column in observed_df, e.g. 'length_change_unc'.
    - modeled_key (str): Key for the modeled values in modeled_df, e.g. 'lengthchange_dLdt_model_array_annual_myr'.
    - item_name (str): Name of the item to be compare, e.g. 'Length Change'.
    - period (str): Period for the data, e.g. '2000-2010'.
    - Xlabel (str): Label for the x-axis. e.g. 'Length Change (m, observed)', 'Climatic Mass Balance (m w.e. a$^{-1}$, observed)'.
    - Ylabel (str): Label for the y-axis. e.g. 'Length Change (m, modeled)', 'Climatic Mass Balance (m w.e. a$^{-1}$, modeled)''.
    - Xlim (tuple): Limits for the x-axis, e.g. (0, 1000).
    - Ylim (tuple): Limits for the y-axis, e.g. (0, 1000).
    - subplot_label_L (str): Label for the left subplot, e.g. 'a)'.
    - subplot_label_R (str): Label for the right subplot, e.g. 'b)'.
    - title (str): Title for the plot.
    - save_path (str): Path to save the plot.
    - save_name (str): Name to save the plot file.
    - legend_index (bool): Whether to include legend index in the plot. The default is True.
    """
    # Merge DataFrames
    merged_df = pd.merge(observed_df, modeled_df, left_on='rgiid', right_on='rgiid', how='inner')

    # Extract data
    # scale the length change from m to km
    if obs_name == 'length_change':
        scale_v = 0.001  # Changed from 0.0001 to properly convert m to km
    else:
        scale_v = 1
    obs_change = merged_df[obs_name].values * scale_v
    obs_unc = merged_df[obs_unc_name].values * scale_v
    model_mean = merged_df['mean'].values * scale_v
    hdi_low, hdi_high = merged_df['hdi_95_low'].values * scale_v, merged_df['hdi_95_high'].values * scale_v

    # Create figure with proper layout management
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    #fig.suptitle(title, fontsize=18, y=1.02)

    # Create gridspec with adjusted margins
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1], wspace=0.17, 
                  left=0.04, right=0.86, bottom=0.15, top=0.9)

    # --- One-to-One Plot with Boxplots (left subplot) ---
    ax1 = fig.add_subplot(gs[0])

    # Create boxplot data for each RGI ID using ensemble results
    box_data = []
    for _, row in modeled_df_raw.iterrows():
        # Convert string representation to numpy array if needed
        if isinstance(row[modeled_key], str):
            # More robust string to array conversion
            ensemble_members = np.array([float(x) for x in row[modeled_key].strip('[]').split()]) * scale_v
        else:
            ensemble_members = np.array(row[modeled_key]) * scale_v
        box_data.append(ensemble_members)  # THIS WAS MISSING IN ORIGINAL CODE

    # Calculate default limits if not provided
    if Xlim is None:
        all_box_values = np.concatenate(box_data)
        Xlim = (min(obs_change.min(), all_box_values.min()), 
                max(obs_change.max(), all_box_values.max()))
    if Ylim is None:
        Ylim = Xlim  # Use same limits for y-axis

    # Position boxes at observed value positions
    box_positions = obs_change
    box_widths = 0.05 * (Xlim[1] - Xlim[0])  # 5% of total x-range

    boxprops = dict(facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=1.5)
    whiskerprops = dict(color='orange', alpha=0.8)
    medianprops = dict(color='orange', linewidth=1.5)
    capprops = dict(color='orange', linewidth=1.5)

    # Ensure we have matching lengths
    if len(box_data) != len(box_positions):
        raise ValueError(f"Length mismatch: box_data ({len(box_data)}) != box_positions ({len(box_positions)})")

    ax1.boxplot(box_data,
               positions=box_positions,
               widths=box_widths,
               patch_artist=True,
               boxprops=boxprops,
               whiskerprops=whiskerprops,
               medianprops=medianprops,
               capprops=capprops,
               showfliers=False,
               manage_ticks=False)

    # Create a proxy artist for the boxplot in the legend
    box_legend = plt.Line2D([0], [0], color='orange', alpha=0.3, lw=5, label='Modeled')

    # Plot values with error bars
    ax1.errorbar(obs_change, model_mean,
               xerr=obs_unc, markersize=4,
               fmt='o', color='blue', alpha=0.7,
               label='Observed', capsize=3)

    # Plot 1:1 line
    ax1.plot(Xlim, Xlim, 'k--', label='1:1 Line')

    # Configure plot
    ax1.set(xlabel=Xlabel,
           ylabel=Ylabel,
           aspect='equal',
           xlim=Xlim,
           ylim=Ylim)
    #ax1.legend(handles=ax1.get_legend_handles_labels()[0] + [box_legend])
    ax1.grid(linestyle='--')
    #set the legend
    if legend_index:
        # If legend_index is True, show the legend including existing handles and box_legend
        handles = ax1.get_legend_handles_labels()[0] + [box_legend]
        ax1.legend(handles=handles, frameon=False)  # Show legend without box edge
    else:
        # If legend_index is False, do not show the legend at all
        ax1.legend().set_visible(False)  # This effectively hides the legend
    # Add 'a)' label outside top-left
    ax1.annotate(subplot_label_L, xy=(-0.13, 0.96), xycoords='axes fraction',
                fontsize=18, weight='normal', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', pad=0))
    # --- CDF Plot (right subplot) ---
    ax2 = fig.add_subplot(gs[1])
    
    # Plot observed CDF and uncertainty
    sns.ecdfplot(obs_change, ax=ax2, label='Observed', color='blue')
    
    # Calculate and plot observed uncertainty band
    lower_obs = obs_change - obs_unc
    upper_obs = obs_change + obs_unc
    x_obs = np.linspace(lower_obs.min(), upper_obs.max(), 1000)
    cdf_lower = np.searchsorted(np.sort(lower_obs), x_obs, side='right') / len(lower_obs)
    cdf_upper = np.searchsorted(np.sort(upper_obs), x_obs, side='right') / len(upper_obs)
    ax2.fill_between(x_obs, cdf_lower, cdf_upper, color='blue', alpha=0.3, label='Observed Uncertainty')
    
    # Plot modeled CDF and HDI
    sns.ecdfplot(model_mean, ax=ax2, label='Modeled Mean', color='orange')
    x_model = np.linspace(hdi_low.min(), hdi_high.max(), 1000)
    cdf_hdi_low = np.searchsorted(np.sort(hdi_low), x_model, side='right') / len(hdi_low)
    cdf_hdi_high = np.searchsorted(np.sort(hdi_high), x_model, side='right') / len(hdi_high)
    ax2.fill_between(x_model, cdf_hdi_low, cdf_hdi_high, color='orange', alpha=0.3, label='HDI 95%')

    ax2.set(xlabel=item_name,
           ylabel='Cumulative Probability',
           xlim = Xlim)

    ax2.grid(linestyle='--')
    #set the legend
    if legend_index:
        # If legend_index is True, show the legend including existing handles
        handles = ax2.get_legend_handles_labels()[0]
        ax2.legend(handles=handles, frameon=False)
    else:
        # If legend_index is False, do not show the legend at all
        ax2.legend().set_visible(False)
    # Add 'b)' label outside top-left
    ax2.annotate(subplot_label_R, xy=(-0.14, 0.96), xycoords='axes fraction',
                fontsize=18, weight='normal', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', pad=0))
    # Adjust layout and save
    plt.tight_layout()
    # Save the figure
    if save_path is None:
        save_path = 'Regional_analysis/figures/'
    else:
        save_path = os.path.join(save_path, 'Regional_analysis', 'figures')
    os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        save_name = 'cdf_and_one_to_one_comparison' + f'_{item_name.replace(" ", "_")}' + f'_{period.replace("-", "_")}'
    save_file = os.path.join(save_path, f"{save_name}.png")
    plt.savefig(save_file, bbox_inches='tight', dpi=300)

    #print(f"Figure saved to {save_file}")

    #plt.show()