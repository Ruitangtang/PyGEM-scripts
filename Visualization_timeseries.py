## Plotting timeseries variables for each glacier
## Ruitang Yang (ruitang.yang@geo.uio.no)
## Last update: 2024-08-29


# Required libraries
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
import os
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D  # Import Line2D for custom legend
import pdb
import matplotlib.cm as cm

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
    date_range_Tick = [d.strftime('%Y-%m') for d in date_range]

    
    # Plotting the data with the new datetime index
    plt.figure(figsize=(12, 6))
    plt.plot(date_range, data , color='blue')
    
    # set x-axis ticks and lables
    plt.xticks(date_range, date_range_Tick, rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel(Y_label if Y_label else 'Variable Name')

        # Adding labels and title
    plt.title(F_title)

    # Adding dashed vertical lines after each year
    years = pd.date_range(start=date_range.min(), end=date_range.max(), freq='YS')  # Year start frequency
    for year in years:
        plt.axvline(x=year, linestyle='--', color='gray', linewidth=0.5)  # Add a dashed vertical line
    
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
                              save_path=None,save_name=None,observation_error=None):
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
        # Adjust the legend: place it outside, split into multiple columns
        #pdb.set_trace()
        if num_models > 5:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small', title= None)
            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for the legend
        else:
            plt.legend()

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
        # Adjust the legend: place it outside, split into multiple columns
        if models_output.shape[0] > 5:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small', title= None)
            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for the legend
        else:
            plt.legend()
        # plt.grid(True)
        # plt.show()

    else:
        raise ValueError("Invalid plot_type. Choose 'point' for point-to-point comparison or 'timeseries' for timeseries comparison.")

    # Save the figure if save_path is provided
    if save_path and save_name:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")


        # Display the plot
        #plt.show()