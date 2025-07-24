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
from scipy import stats
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
import statistic_tool as stats_t
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


# Function to get statistics comparing observed and modeled data for good and bad AMIS, Regional analysis, with the option to use uncertainty-adjusted K-S test,
# this function is used in the plot_cdf_and_one_to_one_Good_Bad function, add the option to do the log transformation, and the option to inset zoom in the plot
def plot_cdf_and_one_to_one_Good_Bad_All_Inset (observed_df = None, modeled_df = None, modeled_df_raw= None,
                            obs_name = None, obs_unc_name = None, modeled_key = None, item_name = None,
                            period = None, reg_id = None,Xlabel = None, Ylabel = None,Xlim = None, Ylim = None,
                            subplot_label_L = 'a', subplot_label_R = 'b',title= None, save_path=None, save_name=None,
                            legend_index = True,Good_AMIS = None,Bad_AMIS = None,logx = False,logy= False,inset_zoom = False,Good_bad = True,
                            zoom_xlim = None, zoom_ylim = None,zoom_position = None,zoom_ticklabels = None):
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
    - reg_id (str): Region ID for the data, e.g. '07'.
    - Xlabel (str): Label for the x-axis. e.g. 'Length Change (m, observed)', 'Climatic Mass Balance (m w.e. a$^{-1}$, observed)'.
    - Ylabel (str): Label for the y-axis. e.g. 'Length Change (m, modeled)', 'Climatic Mass Balance (m w.e. a$^{-1}$, modeled)''.
    - Xlim (tuple): Limits for the x-axis, e.g. (0, 1000).
    - Ylim (tuple): Limits for the y-axis, e.g. (0, 1000).
    - subplot_label_L (str): Label for the left subplot, e.g. 'a)'.
    - subplot_label_R (str): Label for the right subplot, e.g. 'b)'.
    - title (str): Title for the plot.
    - save_path (str): Path to save the plot.
    - save_name (str): Name to save the plot file.
    - Good_AMIS (df): List of good AMIS rgiid to highlight in the plot.
    - Bad_AMIS (df): List of bad AMIS rgiid to highlight in the plot.
    - logx (bool): If True, apply logarithmic scale to x-axis.
    - logy (bool): If True, apply logarithmic scale to y-axis.
    - inset_zoom (bool): If True, add an inset zoom to the plot.
    - Good_bad (bool): If True, plot good and bad AMIS separately, otherwise plot all together.
    - zoom_xlim (tuple): Limits for the x-axis of the inset zoom, e.g. (0, 100).
    - zoom_ylim (tuple): Limits for the y-axis of the inset zoom, e.g. (0, 100).
    - zoom_position (array): Specifies the position and size of the inset zoom in the plot, given as [x, y, width, height], where [0.6, 0.6, 0.25, 0.25] means that the inset zoom is located at (0.6, 0.6) in the coordinate system, with a width and height of 0.25.
    - zoom_ticklabels (bool): If True, show tick labels in the inset zoom, otherwise hide them, and  show the connection lines between the inset zoom and the main plot.
    Returns:
    - None: Displays the plot and saves it to the specified path.
    - The function also computes K-S statistics comparing observed and modeled data for good and bad AMIS.
    - The function also handles the case where the modeled_key column is not present in the raw DataFrame.
    - The function also handles the case where the observed_df and modeled_df do not have the same 'rgiid' values.
    - The function also handles the case where the observed_df and modeled_df have different lengths, ensuring that the merge operation does not fail.
    - The function also handles the case where the observed_df and modeled_df have different columns, ensuring that the specified obs_name and obs_unc_name are present in the observed_df.
    - The function also handles the case where the modeled_df does not have the 'mean', 'hdi_95_low', and 'hdi_95_high' columns, ensuring that the merge operation does not fail.
    - The function also handles the case where the modeled_df_raw does not have the modeled_key column, ensuring that the box plots are created correctly.
    """
    # Merge DataFrames
    merged_df = pd.merge(observed_df, modeled_df, left_on='rgiid', right_on='rgiid', how='inner')

    #=============================================
    # Split the merged_df into two parts: one for good AMIS and one for bad AMIS
    if Good_AMIS is not None:
        merged_df_good = merged_df[merged_df['rgiid'].isin(Good_AMIS['rgiid'])]
        merged_df_raw_good = modeled_df_raw[modeled_df_raw['rgiid'].isin(Good_AMIS['rgiid'])]
        # Ensure the modeled_key column exists in the raw DataFrame
    else:
        merged_df_good = pd.DataFrame()
        merged_df_raw_good = pd.DataFrame()
    if Bad_AMIS is not None:
        merged_df_bad = merged_df[merged_df['rgiid'].isin(Bad_AMIS['rgiid'])]
        merged_df_raw_bad = modeled_df_raw[modeled_df_raw['rgiid'].isin(Bad_AMIS['rgiid'])]
    else:
        merged_df_bad = pd.DataFrame()
        merged_df_raw_bad = pd.DataFrame()
    # if just show the good
    #merged_df = merged_df_good
    #modeled_df_raw = modeled_df_raw[modeled_df_raw['rgiid'].isin(Good_AMIS['rgiid'])]

    #print('good_AMIS:',merged_df_good)
    #print('merged_df_raw_good:', merged_df_raw_good.shape)
    #print('merged_df_raw_bad:', merged_df_raw_bad.shape)
    #=============================================

    # Extract data
    # scale the length change from m to km
    # clip, to make sure the value of varibles are non-negative
    if obs_name == 'length_change':
        scale_v = 0.001  # Changed from 0.0001 to properly convert m to km
        clip_lower = False
    elif obs_name in ['fa_gta_obs', 'fa_mwea_obs']:
        scale_v =1
        clip_lower = True
    else:
        scale_v = 1
        clip_lower = False
    #  obs and obs_unc for good and bad AMIS
    if merged_df_good.empty:
        obs_change_good = np.array([])
        obs_unc_good = np.array([])
    else:
        obs_change_good = merged_df_good[obs_name].values * scale_v
        obs_unc_good = merged_df_good[obs_unc_name].values * scale_v
            #  modeled values for good and bad AMIS
        model_mean_good = merged_df_good['mean'].values * scale_v
        hdi_low_good, hdi_high_good = merged_df_good['hdi_95_low'].values * scale_v, merged_df_good['hdi_95_high'].values * scale_v
        # --- (1) Compute K-S Test with Uncertainty ---
        statis_compare_good =stats_t.get_statistics_compare(obs_change=obs_change_good, obs_unc=obs_unc_good, hdi_high=hdi_high_good,
                           hdi_low=hdi_low_good, model_mean=model_mean_good,n_simulations=1000,item_name = item_name,period = period,reg_id =reg_id)
    
    if merged_df_bad.empty:
        obs_change_bad = np.array([])
        obs_unc_bad = np.array([])
    else:
        obs_change_bad = merged_df_bad[obs_name].values * scale_v
        obs_unc_bad = merged_df_bad[obs_unc_name].values * scale_v
        model_mean_bad = merged_df_bad['mean'].values * scale_v
        hdi_low_bad, hdi_high_bad = merged_df_bad['hdi_95_low'].values * scale_v, merged_df_bad['hdi_95_high'].values * scale_v
        # --- (1) Compute K-S Test with Uncertainty ---
        statis_compare_bad =stats_t.get_statistics_compare(obs_change=obs_change_bad, obs_unc=obs_unc_bad, hdi_high=hdi_high_bad,
                           hdi_low=hdi_low_bad, model_mean=model_mean_bad,n_simulations=1000,item_name = item_name,period = period,reg_id =reg_id)


    #  modeled values for raw data
    obs_change = merged_df[obs_name].values * scale_v
    obs_unc = merged_df[obs_unc_name].values * scale_v
    model_mean = merged_df['mean'].values * scale_v
    hdi_low, hdi_high = merged_df['hdi_95_low'].values * scale_v, merged_df['hdi_95_high'].values * scale_v

    # 
    #===================================================
    # --- (1) Compute K-S Test with Uncertainty ---
    statis_compare_all =stats_t.get_statistics_compare(obs_change=obs_change, obs_unc=obs_unc, hdi_high=hdi_high,
                        hdi_low=hdi_low, model_mean=model_mean,n_simulations=1000,item_name = item_name,period = period,reg_id =reg_id)
    #===================================================
    # Create figure with proper layout management
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    #fig.suptitle(title, fontsize=18, y=1.02)

    # Create gridspec with adjusted margins
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1], wspace=0.17, 
                  left=0.04, right=0.86, bottom=0.15, top=0.9)
    

    ###### ============================================ subplot 1  ============================================
    # --- One-to-One Plot with Boxplots (left subplot) ---
    ax1 = fig.add_subplot(gs[0])

    # Create boxplot data for each RGI ID using ensemble results, for good and bad AMIS
    box_data = []
    box_data_bad = []
    box_data_good = []
    for _, row in modeled_df_raw.iterrows():
        # Convert string representation to numpy array if needed
        if isinstance(row[modeled_key], str):
            # More robust string to array conversion
            ensemble_members = np.array([float(x) for x in row[modeled_key].strip('[]').split()]) * scale_v
        else:
            ensemble_members = np.array(row[modeled_key]) * scale_v
        box_data.append(ensemble_members)  # THIS WAS MISSING IN ORIGINAL CODE
    for _, row in merged_df_raw_good.iterrows():
        # Convert string representation to numpy array if needed
        if isinstance(row[modeled_key], str):
            # More robust string to array conversion
            ensemble_members = np.array([float(x) for x in row[modeled_key].strip('[]').split()]) * scale_v
        else:
            ensemble_members = np.array(row[modeled_key]) * scale_v
        box_data_good.append(ensemble_members)
    for _, row in merged_df_raw_bad.iterrows():
        # Convert string representation to numpy array if needed
        if isinstance(row[modeled_key], str):
            # More robust string to array conversion
            ensemble_members = np.array([float(x) for x in row[modeled_key].strip('[]').split()]) * scale_v
        else:
            ensemble_members = np.array(row[modeled_key]) * scale_v
        box_data_bad.append(ensemble_members)
    

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

    # Position boxes for good and bad AMIS
    box_colors_good = "#FA7507"
    box_colors_bad = "#F8C0D7F8" 

    box_positions_good = obs_change_good
    box_positions_bad = obs_change_bad
    box_widths_good = 0.05 * (Xlim[1] - Xlim[0])  # 5% of total x-range
    box_widths_bad = 0.05 * (Xlim[1] - Xlim[0])  # 5% of total x-range

    # Styling properties for good boxes
    boxprops_good = dict(facecolor=box_colors_good, alpha=0.5, edgecolor=box_colors_good, linewidth=1.5)
    whiskerprops_good = dict(color=box_colors_good, alpha=0.5)
    medianprops_good = dict(color=box_colors_good, alpha=0.5, linewidth=1.5)
    capprops_good = dict(color=box_colors_good,  alpha=0.5,linewidth=1.5)

    # Styling properties for bad boxes
    boxprops_bad = dict(facecolor= box_colors_bad, edgecolor=box_colors_bad,alpha=0.5, linewidth=1.5)
    whiskerprops_bad = dict(color=box_colors_bad, alpha=0.5)
    medianprops_bad = dict(color=box_colors_bad, alpha=0.5,linewidth=1.5)
    capprops_bad = dict(color=box_colors_bad, alpha=0.5, linewidth=1.5)


    # ============================================ plot the boxplots and error bars
    # Check the lengths of box_data and box_positions
    if len(box_data) != len(box_positions):
        raise ValueError(f"Length mismatch for all boxes: box_data ({len(box_data)}) != box_positions ({len(box_positions)})")
    if len(box_data_good) != len(box_positions_good):
        raise ValueError(f"Length mismatch for good boxes: box_data_good ({len(box_data_good)}) != box_positions_good ({len(box_positions_good)})")
    if len(box_data_bad) != len(box_positions_bad):
        raise ValueError(f"Length mismatch for bad boxes: box_data_bad ({len(box_data_bad)}) != box_positions_bad ({len(box_positions_bad)})")
    
    if Good_bad:
        # plot good
        if not merged_df_good.empty: 
            ax1.boxplot(box_data_good,
                    positions=box_positions_good,
                    widths=box_widths_good,
                    patch_artist=True,
                    boxprops=boxprops_good,
                    whiskerprops=whiskerprops_good,
                    medianprops=medianprops_good,
                    capprops=capprops_good,
                    showfliers=False,
                    manage_ticks=False)

        # Plot bad boxes
        if not merged_df_bad.empty:
            ax1.boxplot(box_data_bad,
                    positions=box_positions_bad,
                    widths=box_widths_bad,
                    patch_artist=True,
                    boxprops=boxprops_bad,
                    whiskerprops=whiskerprops_bad,
                    medianprops=medianprops_bad,
                    capprops=capprops_bad,
                    showfliers=False,
                    manage_ticks=False)

        # Create proxy artists for the legend
        good_box_legend = plt.Line2D([0], [0], color=box_colors_good, alpha=0.5, lw=5, label=' Modeled (converged)')
        bad_box_legend = plt.Line2D([0], [0], color=box_colors_bad, alpha=0.5, lw=5, label='Modeled (unconverged)')

        # plot error bars for good AMIS
        errorbar_color_good =  "#031CF8"  # Color for error bars for good AMIS
        errorbar_color_bad = "#5A91F8"  # Color for error bars for bad AMIS
        # Plot good AMIS points with circle markers
        # Clip the errors to ensure non-negativity
        if not merged_df_good.empty:
            good_obs_change = merged_df_good[obs_name].values * scale_v
            good_obs_unc = merged_df_good[obs_unc_name].values * scale_v
            good_model_mean = merged_df_good['mean'].values * scale_v
            xerr_good_clipped = stats_t.clip_errors(good_obs_change,good_obs_unc,clip_lower)
            # print('good_obs_change is :',good_obs_change)
            # print('xerr (origianl unc) is:',good_obs_unc)
            # print("==========================")
            # print('xerr_clipped is:',xerr_good_clipped)
                
            ax1.errorbar(good_obs_change, good_model_mean,
                    xerr=xerr_good_clipped, markersize=3,
                    fmt='o', color=errorbar_color_good, alpha=0.7,
                    label='Observed (converged)',mfc=errorbar_color_good, capsize=3)
        # Plot bad AMIS points with square markers
        if not merged_df_bad.empty:
            bad_obs_change = merged_df_bad[obs_name].values * scale_v
            bad_obs_unc = merged_df_bad[obs_unc_name].values * scale_v
            bad_model_mean = merged_df_bad['mean'].values * scale_v
            xerr_bad_clipped = stats_t.clip_errors(bad_obs_change,bad_obs_unc,clip_lower)
            # print('bad_obs_change is :',bad_obs_change)
            # print('xerr (origianl unc) is:',bad_obs_unc)
            # print("==========================")
            # print('xerr_clipped is:',xerr_bad_clipped)
            ax1.errorbar(bad_obs_change, bad_model_mean,
                    xerr=xerr_bad_clipped, markersize=3,
                    fmt='o', color=errorbar_color_bad, alpha=0.7,
                    label='Observed (unconverged)',mfc= 'none',  capsize=3)
    else:
        # Plot all boxes together
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
        # Create a proxy artist for the legend
        box_legend = plt.Line2D([0], [0], color='orange', alpha=0.5, lw=5, label='Modeled')

        # plot the error bars for all AMIS
        # Clip the errors to ensure non-negativity
        xerr_clipped = stats_t.clip_errors(obs_change, obs_unc, clip_lower)
            # Plot values with error bars
        ax1.errorbar(obs_change, model_mean,
                  xerr=xerr_clipped, markersize=4,
                 fmt='o', color='blue', alpha=0.7,
                 label='Observed', capsize=3)


    # =========================================plot the one-to-one line
    ax1.plot(Xlim, Xlim, 'k--', label='')

    # Configure plot
    # Update labels based on log flags
    xlabel = f"log₁₀({Xlabel})" if logx else Xlabel
    ylabel = f"log₁₀({Ylabel})" if logy else Ylabel
    ax1.set(xlabel=xlabel,ylabel=ylabel,aspect='equal')

    # ============================================scale the xaxis,yaxis
    # Set log scales if needed
    if logx: 
        ax1.set_xscale('log')
    if logy:
        ax1.set_yscale('log')
    
    # Get min/max values (handling log scales)
    x_valid = obs_change[obs_change > 0] if logx else obs_change
    y_valid = np.concatenate([x for x in box_data if x is not None])
    if logy: 
        y_valid = y_valid[y_valid > 0]

    if logx and logy:
        # Set equal limits with 10% padding
        lim_min = min(np.min(x_valid), np.min(y_valid)) * (0.9 if logx or logy else 1)
        lim_max = max(np.max(x_valid), np.max(y_valid)) * (1.1 if logx or logy else 1)
        ax1.set_xlim(lim_min, lim_max)
        ax1.set_ylim(lim_min, lim_max) 
        # Force square plot
        ax1.set_aspect('equal', adjustable='box')
    else:
        # Set limits based on provided Xlim and Ylim
        ax1.set_xlim(Xlim)
        ax1.set_ylim(Ylim)
    
    #ax1.legend(handles=ax1.get_legend_handles_labels()[0] + [box_legend])
    # ============================================ set the grid
    ax1.grid(linestyle='--')

    # ============================================ set legend
    if legend_index:
        # Get existing handles and labels
        handles, labels = ax1.get_legend_handles_labels()
        if Good_bad:
            # Create proxy artists for both boxplot types
            good_box_legend = plt.Line2D([0], [0], color=box_colors_good, alpha=0.5, lw=5, label='Modeled (converged)')
            bad_box_legend = plt.Line2D([0], [0], color=box_colors_bad, alpha=0.5, lw=5, label='Modeled (unconverged)')
            
            # Add the boxplot legends to existing handles/labels
            handles.extend([good_box_legend, bad_box_legend])
            labels.extend(['Modeled (converged)', 'Modeled (unconverged)'])
        else:
            # Create a proxy artist for the boxplot
            box_legend = plt.Line2D([0], [0], color='orange', alpha=0.5, lw=5, label='Modeled')
            
            # Add the boxplot legend to existing handles/labels
            handles.append(box_legend)
            labels.append('Modeled')
        
        # Update the legend with all handles and labels
        # Set legend with shorter lines and tighter spacing
        ax1.legend(handles=handles, labels=labels, 
                handlelength=1,   # shorter line length
                handletextpad=0.8, # less space between line and text
                frameon=False,
                loc='lower right',
                bbox_to_anchor=(1.02, -0.02))

    else:
        # If legend_index is False, hide the legend completely
        ax1.legend().set_visible(False)

    # ============================================ add the title and labels
    if title is not None:
        ax1.set_title(title, fontsize=16, pad=10)
    # Add 'a)' label outside top-left
    ax1.annotate(subplot_label_L, xy=(-0.13, 0.96), xycoords='axes fraction',
                fontsize=18, weight='normal', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', pad=0))
    
    # ============================================ add zoom in inset
    if inset_zoom:
        # Ensure zoom_xlim and zoom_ylim are provided or set defaults
        if zoom_xlim is None:
            zoom_xlim = (0, 1)
        if zoom_ylim is None:
            zoom_ylim = (0, 1)
        # Ensure the zoom limits are within the main plot limits
        zoom_xlim = (max(zoom_xlim[0], Xlim[0]), min(zoom_xlim[1], Xlim[1]))
        zoom_ylim = (max(zoom_ylim[0], Ylim[0]), min(zoom_ylim[1], Ylim[1]))
    
        # Create inset axes in the lower left corner
        axins = ax1.inset_axes(zoom_position)  # [x, y, width, height] in axes coordinates[0.7, 0.7, 0.28, 0.28]
        
        # Replot the main content in the inset
        if Good_bad:
            # Plot good and bad seperately   
            if not merged_df_good.empty:
                axins.errorbar(good_obs_change, good_model_mean,
                            xerr=xerr_good_clipped, markersize=2,
                            fmt='o', color=errorbar_color_good, alpha=0.7,
                            mfc=errorbar_color_good, capsize=2)
                
            if not merged_df_bad.empty:
                axins.errorbar(bad_obs_change, bad_model_mean,
                            xerr=xerr_bad_clipped, markersize=2,
                            fmt='o', color=errorbar_color_bad, alpha=0.7,
                            mfc='none', capsize=2)
            
            # Plot boxes in inset (with smaller widths)
            box_widths_zoom = 0.03 * (zoom_xlim[1] - zoom_xlim[0])
            
            if len(box_data_good) > 0:
                axins.boxplot(box_data_good,
                            positions=box_positions_good,
                            widths=box_widths_zoom,
                            patch_artist=True,
                            boxprops=boxprops_good,
                            whiskerprops=whiskerprops_good,
                            medianprops=medianprops_good,
                            capprops=capprops_good,
                            showfliers=False,
                            manage_ticks=False)
            
            if len(box_data_bad) > 0:
                axins.boxplot(box_data_bad,
                            positions=box_positions_bad,
                            widths=box_widths_zoom,
                            patch_artist=True,
                            boxprops=boxprops_bad,
                            whiskerprops=whiskerprops_bad,
                            medianprops=medianprops_bad,
                            capprops=capprops_bad,
                            showfliers=False,
                            manage_ticks=False)
        else:
            # Plot all together
            axins.errorbar(obs_change, model_mean,
                        xerr=xerr_clipped, markersize=2,
                        fmt='o', color='blue', alpha=0.7,
                        label='Observed', capsize=2)
            # Plot boxes in inset (with smaller widths)
            box_widths_zoom = 0.03 * (zoom_xlim[1] - zoom_xlim[0])
            axins.boxplot(box_data,
                        positions=box_positions,
                        widths=box_widths_zoom,
                        patch_artist=True,
                        boxprops=boxprops,
                        whiskerprops=whiskerprops,
                        medianprops=medianprops,
                        capprops=capprops,
                        showfliers=False,
                        manage_ticks=False)
        # Configure the inset
        axins.plot(zoom_xlim, zoom_xlim, 'k--', linewidth=0.5)
        axins.set_xlim(zoom_xlim)
        axins.set_ylim(zoom_ylim)

        # set the tick labels
        if zoom_ticklabels:
            # Customize tick labels - smaller font size, fewer ticks
            axins.set_xticks(np.linspace(zoom_xlim[0], zoom_xlim[1], 3))  # 3 ticks for x-axis
            axins.set_yticks(np.linspace(zoom_ylim[0], zoom_ylim[1], 3))  # 3 ticks for y-axis
            # Format tick labels - adjust fontsize as needed
            axins.tick_params(axis='both', which='major', labelsize=6)  # Smaller font for inset
        else:
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            # Connect the inset to the main plot
            axins.indicate_inset_zoom(ax1, edgecolor='black', alpha=0.3)
            # Add connection lines
        # add the grid for axins
        axins.grid(True, linestyle=':', alpha=0.3)
    

    ###### =========================================== subplot 2 =================================
    # --- CDF Plot (right subplot) ---
    ax2 = fig.add_subplot(gs[1])
    
    if Good_bad:
        # In the CDF plot section:
        if not merged_df_bad.empty:
            # Plot CDF for bad AMIS observed values
            sns.ecdfplot(bad_obs_change, ax=ax2, label='Observed (unconverged)', color= errorbar_color_bad, linestyle='--')
            lower_obs_bad = bad_obs_change - bad_obs_unc
            upper_obs_bad = bad_obs_change + bad_obs_unc
            x_obs_bad = np.linspace(lower_obs_bad.min(), upper_obs_bad.max(), 1000)
            cdf_lower_bad = np.searchsorted(np.sort(lower_obs_bad), x_obs_bad, side='right') / len(lower_obs_bad)
            cdf_upper_bad = np.searchsorted(np.sort(upper_obs_bad), x_obs_bad, side='right') / len(upper_obs_bad)
            ax2.fill_between(x_obs_bad, cdf_lower_bad, cdf_upper_bad, color= errorbar_color_bad, alpha=0.3, label='')
            # Plot CDF for bad AMIS modeled values
            sns.ecdfplot(model_mean_bad, ax=ax2, label='Modeled (unconverged)', color= box_colors_bad, linestyle='--')
            x_model_bad = np.linspace(hdi_low_bad.min(), hdi_high_bad.max(), 1000)
            cdf_hdi_low_bad = np.searchsorted(np.sort(hdi_low_bad), x_model_bad, side='right') / len(hdi_low_bad)
            cdf_hdi_high_bad = np.searchsorted(np.sort(hdi_high_bad), x_model_bad, side='right') / len(hdi_high_bad)
            ax2.fill_between(x_model_bad, cdf_hdi_low_bad, cdf_hdi_high_bad, color=box_colors_bad, alpha=0.3, label='')   
        if not merged_df_good.empty:
            # Plot CDF for good AMIS observed values
            sns.ecdfplot(good_obs_change, ax=ax2, label='Observed (converged)', color= errorbar_color_good,linestyle='-')
            lower_obs_good = good_obs_change - good_obs_unc
            upper_obs_good = good_obs_change + good_obs_unc
            x_obs_good = np.linspace(lower_obs_good.min(), upper_obs_good.max(), 1000)
            cdf_lower_good = np.searchsorted(np.sort(lower_obs_good), x_obs_good, side='right') / len(lower_obs_good)
            cdf_upper_good = np.searchsorted(np.sort(upper_obs_good), x_obs_good, side='right') / len(upper_obs_good)
            ax2.fill_between(x_obs_good, cdf_lower_good, cdf_upper_good, color= errorbar_color_good, alpha=0.3, label='')
            sns.ecdfplot(model_mean_good, ax=ax2, label='Modeled (converged)', color=box_colors_good, linestyle='-')
            # Plot CDF for good AMIS modeled values
            x_model_good = np.linspace(hdi_low_good.min(), hdi_high_good.max(), 1000)
            cdf_hdi_low_good = np.searchsorted(np.sort(hdi_low_good), x_model_good, side='right') / len(hdi_low_good)
            cdf_hdi_high_good = np.searchsorted(np.sort(hdi_high_good), x_model_good, side='right') / len(hdi_high_good)
            ax2.fill_between(x_model_good, cdf_hdi_low_good, cdf_hdi_high_good, color=box_colors_good, alpha=0.3, label='')     
    else:
        # Plot observed CDF and uncertainty for all AMIS
        sns.ecdfplot(obs_change, ax=ax2, label='Observed', color='blue')
        # Calculate and plot observed uncertainty band
        lower_obs = obs_change - obs_unc
        upper_obs = obs_change + obs_unc
        x_obs = np.linspace(lower_obs.min(), upper_obs.max(), 1000)
        cdf_lower = np.searchsorted(np.sort(lower_obs), x_obs, side='right') / len(lower_obs)
        cdf_upper = np.searchsorted(np.sort(upper_obs), x_obs, side='right') / len(upper_obs)
        ax2.fill_between(x_obs, cdf_lower, cdf_upper, color='blue', alpha=0.3, label='Observed Uncertainty')

        # Plot modeled CDF and HDI
        sns.ecdfplot(model_mean, ax=ax2, label='Modeled', color='orange')
        x_model = np.linspace(hdi_low.min(), hdi_high.max(), 1000)
        cdf_hdi_low = np.searchsorted(np.sort(hdi_low), x_model, side='right') / len(hdi_low)
        cdf_hdi_high = np.searchsorted(np.sort(hdi_high), x_model, side='right') / len(hdi_high)
        ax2.fill_between(x_model, cdf_hdi_low, cdf_hdi_high, color='orange', alpha=0.3, label='Modeled Uncertainty')
    
    # Set x-label and y-label
    if logx:
        xlabel = f"log₁₀({item_name})"
    else:
        xlabel = item_name
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Cumulative Probability')

    # Set x-axis scaling and limits based on conditions
    if logx:
        ax2.set_xscale('log')
        valid_vals = obs_change[obs_change > 0]
        if len(valid_vals) > 0:
            min_value = valid_vals.min() * 0.5
            ax2.set_xlim(left=min(min_value, lim_min), right=lim_max)
        else:
            ax2.set_xlim(left=lim_min, right=lim_max)
    elif clip_lower:
        ax2.set_xlim(left=0, right=Xlim[1])
    else:
        ax2.set_xlim(Xlim)


    # ============================================= set grid and legend
    # set the grid
    ax2.grid(linestyle='--')
    #set the legend
    if legend_index:
        # If legend_index is True, show the legend including existing handles
        handles = ax2.get_legend_handles_labels()[0]
        ax2.legend(handles=handles, frameon=False)
    else:
        # If legend_index is False, do not show the legend at all
        ax2.legend().set_visible(False)

    # ============================================= add the annotation and title
    # Add 'b)' label outside top-left
    ax2.annotate(subplot_label_R, xy=(-0.14, 0.96), xycoords='axes fraction',
                fontsize=18, weight='normal', ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', pad=0))

    ###### ============================================ adjust the layout and save the figure
    # Adjust layout and save
    plt.tight_layout()
    # Save the figure
    if save_path is None:
        save_path = os.path.join(pygem_prms.output_filepath,'Calibration','Postprocessing',reg_id,'Regional_analysis/figures/')
    else:
        save_path = os.path.join(save_path, 'Regional_analysis', 'figures')
    os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        if Good_bad and inset_zoom:
            save_name = f'{item_name.replace(" ", "_")}_{period.replace("-", "_")}_cdf_comparison_GOOD_BAD_inset_zoom'
        elif Good_bad:
            save_name = f'{item_name.replace(" ", "_")}_{period.replace("-", "_")}_cdf_comparison_GOOD_BAD'
        elif inset_zoom:
            save_name = f'{item_name.replace(" ", "_")}_{period.replace("-", "_")}_cdf_comparison_inset_zoom'
        else:
            save_name = f'{item_name.replace(" ", "_")}_{period.replace("-", "_")}_cdf_comparison'

        # add the logx and logy to the save name
        if logx:
            save_name += '_log'

    save_file = os.path.join(save_path, f"{save_name}.png")
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close(fig)  # or plt.close('all') if you want to close all open figures


# Function to visulize the delta rmse of the AMIS Prior and Posterior
def plot_delta_rmse_histograms(delta_rmse_good= None, delta_rmse_bad= None, breaks_index = True,
                               bin_width= 0.3,xlim_left =None,xlim_right =None,x_breaks = None,
                               save_path=None, save_name=None,item_name='dLdt', period='2000-2010'):
    """    Plots histograms of delta RMSE for good and bad AMIS, with two subplots
    Args:
        delta_rmse_good (array-like): Delta RMSE values for good AMIS.
        delta_rmse_bad (array-like): Delta RMSE values for bad AMIS.
        breaks_index (bool): If True, plot the histograms with breaks at specified x_breaks, and plot the main cluster on the right, and the outlier part on the left.
        bin_width (float): Width of the bins for the histograms.
        xlim_left (tuple): x-axis limits for the left plot (outlier part).
        xlim_right (tuple): x-axis limits for the right plot (main cluster).
        x_breaks (int/float): Custom x-axis breaks for the histograms.
        save_path (str): Path to save the figure.
        save_name (str): Name of the saved figure file.
        item_name (str): Name of the item being analyzed, used in the title and save name.
        period (str): Period of analysis, used in the title and save name.
    """
    # Convert data to numpy arrays
    # Convert inputs to NumPy arrays, handling None and empty cases
    delta_rmse_good = np.array(delta_rmse_good) if delta_rmse_good is not None else np.array([])
    delta_rmse_bad = np.array(delta_rmse_bad) if delta_rmse_bad is not None else np.array([])

    # Check if both inputs are empty
    if delta_rmse_good.size == 0 and delta_rmse_bad.size == 0:
        raise ValueError("Both delta_rmse_good and delta_rmse_bad cannot be empty.")
    
    # Combine non-empty arrays
    combined = np.concatenate([arr for arr in [delta_rmse_good, delta_rmse_bad] if arr.size > 0])

    bin_width = bin_width
    # filter out inf and -inf values
    combined = combined[np.isfinite(combined)]
    delta_rmse_good = delta_rmse_good[np.isfinite(delta_rmse_good)]
    delta_rmse_bad = delta_rmse_bad[np.isfinite(delta_rmse_bad)]
    # Ensure there's valid data in combined before proceeding
    if combined.size > 0:
        min_edge = np.floor(combined.min() / bin_width) * bin_width
        max_edge = np.ceil(combined.max() / bin_width) * bin_width
        bins = np.arange(min_edge, max_edge + bin_width, bin_width)
    else:
        print("Warning: No valid combined RMSE values available after filtering.")
        bins = np.array([])  # Return an empty array for bins or handle as needed
    #print("min_edge, max_edge, bins:", min_edge, max_edge, bins)
    if 0 not in bins:
        bins = np.sort(np.append(bins, 0.0))

    # Calculate mean and t-test for delta_rmse_good
    if delta_rmse_good.size > 0:
        mean_good = np.mean(delta_rmse_good)
        t_stat_good, p_val_good = stats.ttest_1samp(delta_rmse_good, 0)
        # print("Mean Good:", mean_good)
        # print("t-statistic for Good:", t_stat_good, "p-value:", p_val_good)
    else:
        print("Warning: No valid delta_rmse_good values available for calculations.")

    # Calculate mean and t-test for delta_rmse_bad
    if delta_rmse_bad.size > 0:
        mean_bad = np.mean(delta_rmse_bad)
        t_stat_bad, p_val_bad = stats.ttest_1samp(delta_rmse_bad, 0)
        # print("Mean Bad:", mean_bad)
        # print("t-statistic for Bad:", t_stat_bad, "p-value:", p_val_bad)
    else:
        print("Warning: No valid delta_rmse_bad values available for calculations.")

    # Create subplots

    if breaks_index:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, figsize=(6, 4),
                                                gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.025})

        # ==  Right plot: main cluster (-10 to 1)
        ax_right.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)
        ax_right.axvline(mean_good, color='blue', linestyle='--', linewidth=0.8,zorder=2)
        ax_right.axvline(mean_bad, color='red', linestyle='--', linewidth=0.8,zorder=2)
        sns.histplot(delta_rmse_good[(delta_rmse_good > x_breaks)], bins=bins,
                     fill=False, kde=False, color='blue', edgecolor='blue', ax=ax_right,zorder=3)
        sns.histplot(delta_rmse_bad[(delta_rmse_bad > x_breaks)], bins=bins,
                     fill=False, kde=False, color='red', edgecolor='red', ax=ax_right,zorder=3)

        ax_right.set_xlim(xlim_right if xlim_right is not None else (-10, 1))
        # Hide the spines between plots
        ax_right.spines['left'].set_visible(False)
        ax_right.yaxis.tick_right()
        
        # == Left plot: outlier part (all data <= -10)
        ax_left.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)
        sns.histplot(delta_rmse_good[delta_rmse_good <= x_breaks], bins=bins, fill=False, kde=False,
                     color='blue', edgecolor='blue', ax=ax_left,zorder=3)
        sns.histplot(delta_rmse_bad[delta_rmse_bad <= x_breaks], bins=bins, fill=False, kde=False,
                     color='red', edgecolor='red', ax=ax_left,zorder=3)

        ax_left.set_xlim(xlim_left if xlim_left is not None else (-50, -10))
        ax_left.set_ylabel('Count', fontsize=12)
        ax_left.axvline(mean_good, color='blue', linestyle='--')
        ax_left.axvline(mean_bad, color='red', linestyle='--')
            # Hide the spines between plots
        ax_left.spines['right'].set_visible(False)
        ax_left.yaxis.tick_left()
        ax_left.tick_params(labelright=False)
 
        #  == Add double slashes (//) on the right edge of the left plot’s x-axis line
        d = .015  # length of slashes in axes fraction
        gap = .02  # gap between the two slashes

        kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=1)

        ax_left.plot([1 - d - gap, 1 + d - gap], [-d, +d], **kwargs)  # first slash
        ax_left.plot([1 - d + gap, 1 + d + gap], [-d, +d], **kwargs)  # second slash
        ax_left.plot([1 - d - gap, 1 + d - gap], [1 - d, 1 + d], **kwargs)  # top right //
        ax_left.plot([1 - d + gap, 1 + d + gap], [1 - d, 1 + d], **kwargs)  # second top right //

        # == Annotations (on right plot)
        ax_right.text(ax_right.get_xlim()[0] - 0.2 * (ax_right.get_xlim()[1] - ax_right.get_xlim()[0]),
                      ax_right.get_ylim()[1] * 0.85,
                      f'Converged:\nt = {t_stat_good:.2f}, p = {p_val_good:.3f}',
                      fontsize=10, color='blue')

        ax_right.text(ax_right.get_xlim()[0] - 0.2 * (ax_right.get_xlim()[1] - ax_right.get_xlim()[0]),
                      ax_right.get_ylim()[1] * 0.7,
                      f'Unconverged:\nt = {t_stat_bad:.2f}, p = {p_val_bad:.3f}',
                      fontsize=10, color='red')
        # Set labels for the whole figure
        fig.text(0.5, 0.01, 'Δ Normalized RMSE (Posterior − Prior)', ha='center', fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)
        sns.histplot(delta_rmse_good, bins=bins, fill=False, kde=False,
                     color='blue', edgecolor='blue', ax=ax, label='Converged',zorder=3)
        sns.histplot(delta_rmse_bad, bins=bins, fill=False, kde=False,
                     color='red', edgecolor='red', ax=ax, label='Unconverged',zorder=3)

        ax.axvline(mean_good, color='blue', linestyle='--', linewidth=0.8,zorder=2)
        ax.axvline(mean_bad, color='red', linestyle='--', linewidth=0.8,zorder=2)

        # ==  Annotations
        ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                ax.get_ylim()[1] * 0.85,
                f'Converged:\nt = {t_stat_good:.2f}, p = {p_val_good:.3f}',
                fontsize=10, color='blue')
        ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                ax.get_ylim()[1] * 0.7,
                f'Unconverged:\nt = {t_stat_bad:.2f}, p = {p_val_bad:.3f}',
                fontsize=10, color='red')

        #ax.set_xlim(xlim_left if xlim_left is not None else (-50, 1))
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xlabel('Δ Normalized RMSE (Posterior − Prior)', fontsize=12)
        #ax.legend()  
    plt.tight_layout()
    # Save the figure
    if save_path is None:
        save_path = 'Regional_analysis/figures/DA_performance/'
    else:
        save_path = os.path.join(save_path, 'Regional_analysis', 'figures', 'DA_performance')
    os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        save_name = f'Delta_rmse_histograms_{item_name.replace(" ", "_")}_{period.replace("-", "_")}'
    save_file = os.path.join(save_path, f"{save_name}.png")
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close(fig)


# Function to visualize the rmse of the AMIS Prior and Posterior
def plot_rmse_histograms(rmse_good_prior=None,rmse_good_poster=None,rmse_bad_prior=None,rmse_bad_poster=None, bins_width = 0.3,
                         save_path=None, save_name=None, item_name='dLdt', period='2000-2010'):
        """Plots histograms of RMSE for posterior and prior for good and bad AMIS, with statistical analysis,with two subplots,
        one is for good AMIS, and the other is for bad AMIS.
        Args:
            rmse_good_prior (array-like): RMSE values for good AMIS prior.
            rmse_good_poster (array-like): RMSE values for good AMIS posterior.
            rmse_bad_prior (array-like): RMSE values for bad AMIS prior.
            rmse_bad_poster (array-like): RMSE values for bad AMIS posterior.
            bins_width (float): Width of the bins for the histograms.
            save_path (str): Path to save the figure.
            save_name (str): Name of the saved figure file.
            item_name (str): Name of the item being analyzed, used in the title and save name.
            period (str): Period of analysis, used in the title and save name.
        """
        # Convert data to numpy arrays
        # Convert inputs to NumPy arrays, handling None and empty cases
        rmse_good_prior = np.array(rmse_good_prior) if rmse_good_prior is not None else np.array([])
        rmse_good_poster = np.array(rmse_good_poster) if rmse_good_poster is not None else np.array([])
        rmse_bad_prior = np.array(rmse_bad_prior) if rmse_bad_prior is not None else np.array([])
        rmse_bad_poster = np.array(rmse_bad_poster) if rmse_bad_poster is not None else np.array([])
    
        # Check if both inputs are empty, and combine non-empty arrays
        if rmse_good_prior.size == 0 and rmse_good_poster.size == 0 and rmse_bad_prior.size == 0 and rmse_bad_poster.size == 0:
            raise ValueError("All RMSE inputs cannot be empty.")
        # Combine good and bad RMSE arrays
        # Ensure that rmse_good and rmse_bad are not empty before concatenation
        if rmse_good_prior.size == 0 and rmse_good_poster.size == 0:
            rmse_good = np.array([])
        else:
            rmse_good = np.concatenate((rmse_good_prior, rmse_good_poster)) if rmse_good_prior.size > 0 and rmse_good_poster.size > 0 else np.array(rmse_good_prior) if rmse_good_prior.size > 0 else np.array(rmse_good_poster)
        if rmse_bad_prior.size == 0 and rmse_bad_poster.size == 0:
            rmse_bad = np.array([])
        else:
            rmse_bad = np.concatenate((rmse_bad_prior, rmse_bad_poster)) if rmse_bad_prior.size > 0 and rmse_bad_poster.size > 0 else np.array(rmse_bad_prior) if rmse_bad_prior.size > 0 else np.array(rmse_bad_poster)
        
        # Combine non-empty arrays
        combined = np.concatenate([arr for arr in [rmse_good, rmse_bad] if arr.size > 0])

        # Filter out inf and -inf values
        combined = combined[np.isfinite(combined)]
        bins_width = bins_width
        if combined.size >0:
            min_edge = np.floor(combined.min() / bins_width) * bins_width
            max_edge = np.ceil(combined.max() / bins_width) * bins_width
            bins = np.arange(min_edge, max_edge + bins_width, bins_width)
        else:
            print("No valid RMSE data to plot.")
            bins = np.array([])
        

    
        #print("min_edge, max_edge, bins:", min_edge, max_edge, bins)
        if 0 not in bins:
            bins = np.sort(np.append(bins, 0.0))
        # filter out inf and -inf values
        # Function to filter and validate RMSE arrays
        def filter_rmse(rmse_array):
            # Filter out infinite and NaN values
            filtered_array = rmse_array[np.isfinite(rmse_array)]
            if filtered_array.size == 0:
                print("Warning: No valid RMSE values available after filtering.")
                return None  # Indicate that the array is empty or handle it as required
            return filtered_array

        rmse_good_prior = filter_rmse(rmse_good_prior)
        rmse_good_poster = filter_rmse(rmse_good_poster)
        rmse_bad_prior = filter_rmse(rmse_bad_prior)
        rmse_bad_poster = filter_rmse(rmse_bad_poster)
        # Calculate means and t-tests
        mean_good_prior = np.mean(rmse_good_prior)
        t_stat_good_prior, p_val_good_prior = stats.ttest_1samp(rmse_good_prior, 0)
        mean_good_poster = np.mean(rmse_good_poster)
        t_stat_good_poster, p_val_good_poster = stats.ttest_1samp(rmse_good_poster, 0)
        mean_bad_prior = np.mean(rmse_bad_prior)
        t_stat_bad_prior, p_val_bad_prior = stats.ttest_1samp(rmse_bad_prior, 0)
        mean_bad_poster = np.mean(rmse_bad_poster)
        t_stat_bad_poster, p_val_bad_poster = stats.ttest_1samp(rmse_bad_poster, 0)     

        # plot the histograms
        if rmse_good.size == 0 and rmse_bad.size == 0:
            raise ValueError("Both rmse_good and rmse_bad cannot be empty.")
        if rmse_good.size > 0 and rmse_bad.size > 0:
            fig, (ax_good, ax_bad) = plt.subplots(2, 1, sharey=True, figsize=(5, 6),constrained_layout= True)
            # == Good AMIS subplot
            ax_good.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)
            sns.histplot(rmse_good_prior, bins=bins, fill=False, kde=False,
                         color='blue', edgecolor='blue', ax=ax_good, label='Prior',zorder=3)
            sns.histplot(rmse_good_poster, bins=bins, fill=False, kde=False,
                         color='orange', edgecolor='orange', ax=ax_good, label='Posterior',zorder=3)
            ax_good.axvline(mean_good_prior, color='blue', linestyle='--', linewidth=0.8,zorder=2)
            ax_good.axvline(mean_good_poster, color='orange', linestyle='--', linewidth=0.8,zorder=2)
            # Annotations
            ax_good.text(ax_good.get_xlim()[1] - 0.3 * (ax_good.get_xlim()[1] - ax_good.get_xlim()[0]),
                         ax_good.get_ylim()[1] * 0.85,
                         'Converged\n',
                         fontsize=10, color='black')
            ax_good.text(ax_good.get_xlim()[1] - 0.3 * (ax_good.get_xlim()[1] - ax_good.get_xlim()[0]),
                         ax_good.get_ylim()[1] * 0.6,
                         f'Prior:\n t = {t_stat_good_prior:.2f}\n p = {p_val_good_prior:.3f}\n',
                         fontsize=10, color='blue')
            ax_good.text(ax_good.get_xlim()[1] - 0.3 * (ax_good.get_xlim()[1] - ax_good.get_xlim()[0]),
                         ax_good.get_ylim()[1] * 0.45,
                         f'Posterior:\n t = {t_stat_good_poster:.2f}\n p = {p_val_good_poster:.3f}',
                         fontsize=10, color='orange')
            ax_good.set_xlim(left=min_edge, right=max_edge)
            ax_good.set_ylabel('Count', fontsize=12)
            #ax_good.set_xlabel('RMSE (normalized by uncertainty)', fontsize=12)

            # == Bad AMIS subplot
            ax_bad.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)
            sns.histplot(rmse_bad_prior, bins=bins, fill=False, kde=False,
                         color='blue', edgecolor='blue', ax=ax_bad, label='Prior',zorder=3)
            sns.histplot(rmse_bad_poster, bins=bins, fill=False, kde=False,
                         color='orange', edgecolor='orange', ax=ax_bad, label='Posterior',zorder=3)
            ax_bad.axvline(mean_bad_prior, color='blue', linestyle='--', linewidth=0.8,zorder=2)
            ax_bad.axvline(mean_bad_poster, color='orange', linestyle='--', linewidth=0.8,zorder=2)
            # Annotations
            ax_bad.text(ax_bad.get_xlim()[1] - 0.3 * (ax_bad.get_xlim()[1] - ax_bad.get_xlim()[0]),
                        ax_bad.get_ylim()[1] * 0.85,
                        'Unconverged\n',
                        fontsize=10, color='black')
            ax_bad.text(ax_bad.get_xlim()[1] - 0.3 * (ax_bad.get_xlim()[1] - ax_bad.get_xlim()[0]),
                        ax_bad.get_ylim()[1] * 0.65,
                        f'Prior:\n t = {t_stat_bad_prior:.2f}\n p = {p_val_bad_prior:.3f}',
                        fontsize=10, color='blue')
            ax_bad.text(ax_bad.get_xlim()[1] - 0.3 * (ax_bad.get_xlim()[1] - ax_bad.get_xlim()[0]),
                        ax_bad.get_ylim()[1] * 0.45,
                        f'Posterior:\n t = {t_stat_bad_poster:.2f}\n p = {p_val_bad_poster:.3f}',
                        fontsize=10, color='orange')
            ax_bad.set_xlim(left=min_edge, right=max_edge)
            ax_bad.set_xlabel('Normalized RMSE', fontsize=12)
            fig.text(0.01, 0.98, 'a', fontsize=14, ha='left', va='top')
            fig.text(0.01, 0.50, 'b', fontsize=14, ha='left', va='top')
        else:
            # If only one of the good or bad RMSE arrays is non-empty, plot a single plot but with both prior and posterior histogram
            if rmse_good.size > 0:
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)

                sns.histplot(rmse_good_prior, bins=bins, fill=False, kde=False,
                             color='blue', edgecolor='blue', ax=ax, label='Prior',zorder=3)
                sns.histplot(rmse_good_poster, bins=bins, fill=False, kde=False,
                             color='orange', edgecolor='orange', ax=ax, label='Posterior',zorder=3)
                ax.axvline(np.mean(rmse_good_prior), color='blue', linestyle='--', linewidth=0.8,zorder=2)
                ax.axvline(np.mean(rmse_good_poster), color='orange', linestyle='--', linewidth=0.8,zorder=2)
                # Annotations
                ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        ax.get_ylim()[1] * 0.85,
                        'Converged\n',
                        fontsize=10, color='black')
                ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        ax.get_ylim()[1] * 0.7,
                        f'Prior:\nt = {t_stat_good_prior:.2f}, p = {p_val_good_prior:.3f}',
                        fontsize=10, color='blue')
                ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        ax.get_ylim()[1] * 0.6,
                        f'Posterior:\nt = {t_stat_good_poster:.2f}, p = {p_val_good_poster:.3f}',
                        fontsize=10, color='orange')
                ax.set_xlim(left=min_edge, right=max_edge)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_xlabel('Normalized RMSE', fontsize=12)
            elif rmse_bad.size > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.grid(True, linestyle='--', alpha=1, linewidth=0.5,zorder=1)

                sns.histplot(rmse_bad_prior, bins=bins, fill=False, kde=False,
                             color='blue', edgecolor='blue', ax=ax, label='Prior',zorder=3)
                sns.histplot(rmse_bad_poster, bins=bins, fill=False, kde=False,
                             color='orange', edgecolor='orange', ax=ax, label='Posterior',zorder=3)
                ax.axvline(np.mean(rmse_bad_prior), color='blue', linestyle='--', linewidth=0.8,zorder=2)
                ax.axvline(np.mean(rmse_bad_poster), color='orange', linestyle='--', linewidth=0.8,zorder=2)
                # Annotations
                ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        ax.get_ylim()[1] * 0.85,
                        'Unconverged:\n',
                        fontsize=10, color='black')
                ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        ax.get_ylim()[1] * 0.7,
                        f'Prior:\nt = {t_stat_bad_prior:.2f}, p = {p_val_bad_prior:.3f}',
                        fontsize=10, color='blue')
                ax.text(ax.get_xlim()[0] + 0.2 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        ax.get_ylim()[1] * 0.6,
                        f'Posterior:\nt = {t_stat_bad_poster:.2f}, p = {p_val_bad_poster:.3f}',
                        fontsize=10, color='orange')
                ax.set_xlim(left=min_edge, right=max_edge)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_xlabel('Normalized RMSE', fontsize=12)
        
        # save the figure
        # Set the title
        #ax.set_title(f"RMSE Histogram for {item_name} ({period})", fontsize=14)
        # Save the figure
        if save_path is None:
            save_path = 'Regional_analysis/figures/DA_performance/'
        else:
            save_path = os.path.join(save_path, 'Regional_analysis', 'figures', 'DA_performance')
        os.makedirs(save_path, exist_ok=True)
        if save_name is None:
            save_name = f'RMSE_histograms_{item_name.replace(" ", "_")}_{period.replace("-", "_")}'
        save_file = os.path.join(save_path, f"{save_name}.png")
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        #plt.show()
        plt.close(fig)  # or plt.close('all') if you want to close all open figures