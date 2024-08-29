## Plotting timeseries variables for each glacier
## Ruitang Yang (ruitang.yang@geo.uio.no)
## Last update: 2024-08-29


# Required libraries
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import seaborn as sns

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
    plt.xlabel(Y_label if Y_label else 'Variable Name')
    plt.ylabel(Y_label)

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
    plt.title('Calving Flux Time Series')
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
                         title='Flowline profile', save_path=None,save_name =None,xlabel='Distance along the flowline (m)'):
    """
    Plots elevation bands from a NetCDF dataset using xarray and optionally saves the figure.

    Parameters:
    - gdir (GlacierDirectory): The glacier directory object containing the dataset.
    - filesuffix (str): The file suffix identifier for the specific diagnostics file.
    - sel_years (list or array-like): The years to select for plotting the thickness.
    - group (str, optional): The group within the NetCDF file to read data from. Default is 'fl_0'.
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
            fig, ax = plt.subplots(figsize=(10, 6))

        # if the selected years are not provided, plot all years
        if sel_years is None:
            sel_years = ds.time
           

        # Plot bed elevation as a baseline
        ds.bed_h.plot(ax=ax, color='black', label='Bed elevation')

        # Generate a color palette from Seaborn
        colors = sns.color_palette('rocket', len(sel_years))  


        # Plot each year with the gradient colors
        # # Plot the bed height plus thickness for the selected years
        # (ds.bed_h + ds.sel(time=sel_years).thickness_m).plot(ax=ax, hue='time')
        for i, year in enumerate(sel_years):
            (ds.bed_h + ds.sel(time=year).thickness_m).plot(ax=ax, color=colors[i], label=str(year))
  
        # Add a legend to show which colors correspond to which years
        ax.legend(loc='upper left', title='Year')

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
def plot_time_series_snapshots(gdir, filesuffix='', sel_times=None, variable='thickness_m', group='fl_0', 
                               ylabel='Variable Value', xlabel='Distance along the flowline (m)', title='Time Series Snapshots', 
                               save_path=None,save_name=None):
    """
    Plots snapshots of a time series variable from a NetCDF dataset using xarray.

    Parameters:
    - gdir (GlacierDirectory): The glacier directory object containing the dataset.
    - filesuffix (str): The file suffix identifier for the specific diagnostics file.
    - sel_times (list or array-like): The time points to select for plotting the variable.
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
        # Set up a grid of subplots
        num_snapshots = len(sel_times)
        fig, axes = plt.subplots(nrows=1, ncols=num_snapshots, figsize=(5*num_snapshots, 6), sharey=True)

        # Loop through selected time points and plot snapshots
        for i, time_point in enumerate(sel_times):
            # Select data at the specified time
            data_at_time = ds.sel(time=time_point)[variable]

            # Plot on the corresponding axis
            ax = axes[i] if num_snapshots > 1 else axes
            data_at_time.plot(ax=ax, label=f'Time: {time_point}', marker='o')

            # Set labels and title
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(f'{title} - {time_point}')

            # Add legend
            ax.legend()

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

def animate_time_series(ds, variable='thickness_m', interval=200, ylabel='Elevation (m a.s.l.)', 
                        xlabel='Distance (m)', title='Elevation Changes Over Time', save_path=None,save_name=None):
    """
    Creates an animation of a time series variable from a NetCDF dataset using xarray.

    Parameters:
    - ds (xarray Dataset): The dataset containing the variable to plot.
    - variable (str, optional): The variable name in the dataset to animate. Default is 'thickness_m'.
    - interval (int, optional): Delay between frames in milliseconds. Default is 200.
    - ylabel (str, optional): The label for the y-axis. Default is 'Elevation (m a.s.l.)'.
    - xlabel (str, optional): The label for the x-axis. Default is 'Distance (m)'.
    - title (str, optional): The title of the animation plot.
    - save_path (str, optional): The file path to save the animation. If None, the animation will not be saved.

    Returns:
    - None: Displays the animation and optionally saves it.
    """
    # Extract the time points and distances
    times = ds['time'].values
    distance = ds['distance'].values
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', marker='o')
    
    # Set axis labels and title
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    # Initialize the plot limits
    ax.set_xlim(distance.min(), distance.max())
    ax.set_ylim(ds[variable].min(), ds[variable].max())
    
    def update(frame):
        # Update the line data for the current frame (time point)
        time_point = times[frame]
        data_at_time = ds.sel(time=time_point)[variable]
        line.set_data(distance, data_at_time)
        ax.set_title(f'{title} - {np.datetime_as_string(time_point, unit="Y")}')
        return line,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)

    # Save the animation if a save_path is provided
    if save_path:
    # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_full = os.path.join(save_path, save_name)
        plt.savefig(save_path_full, bbox_inches='tight')
        print(f"Figure saved to {save_path_full}")

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