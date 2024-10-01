# load the necessary packages 

import os
import scipy
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
cartopy.config['data_dir'] = os.getenv('CARTOPY_DIR', cartopy.config.get('data_dir'))

import plotly.express as px
import geopandas as gpd
from scipy.spatial import cKDTree

import imageio
import os
import io
import requests

import datetime as dt
from dateutil.relativedelta import *

import netCDF4
from netCDF4 import Dataset, num2date

import cmocean

import seaborn as sns
import pymannkendall as mk

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib import pyplot as plt
# %matplotlib inline

from PIL import Image  # To create gifs
from shapely.geometry import Point
from shapely.geometry import LineString



# Creating a function that will generate the Data Frames for March 2016, 2017, 2018
def makeDataframe(year):

    # Re-adjusting the biogeochemical dataset to match 2017
    month = 3
    netCDF_datestr = str(year)+'-'+format(month, '02')
    
    # GBR4 BIO
    biofiles = ["https://thredds.ereefs.aims.gov.au/thredds/dodsC/GBR4_H2p0_B3p1_Cq3b_Dhnd/daily.nc"]
    
    ds_bio = xr.open_mfdataset(biofiles)
    
    # Define the time range (for the duration of the cycone only)
    start_date = str(year)+'-03-01'
    end_date = str(year)+'-03-31'
    
    # Slice the dataset based on the time range
    ds_bio_slice = ds_bio.sel(time=slice(start_date, end_date), k=-1)
    
    # creating a dataset with only temperature, chlorophyll-a and secchi for the 10 days of the cyclone 
    
    # Select the variables of interest
    selected_vars = ds_bio_slice[['Chl_a_sum', 'temp', 'Secchi']]
    
    # Create a new dataset with these variables
    new_bio = xr.Dataset(
        {
            'Chl_a_sum': selected_vars['Chl_a_sum'],
            'temp': selected_vars['temp'],
            'Secchi': selected_vars['Secchi']
        }
    )

    # Extracting the lon lat of the data
    lon=new_bio.longitude.values
    lat=new_bio.latitude.values
    
    # Build the mesh based on lon lat
    mlon, mlat = np.meshgrid(lon,lat)
    mlonlat = np.dstack([mlon.flatten(), mlat.flatten()])[0]

    # The shape of 'mlon' and 'mlat' is the same as the shape of the new_bio dataset 
    # we create a Numpy list 'mlonlat' that is flattened for the cKDTree

    # mlon.shape        # (723,491)
    # mlonlat.shape     # (354993, 2)
    
    # The shape of 'mlon' and 'mlat' is the same as the shape of the new_bio dataset 
    # we create a Numpy list 'mlonlat' that is flattened for the cKDTree

    # Now, we create the KD-tree based on the new_bio mech, and we add the cyclone path 
    # Building the cKDtree
    tree = scipy.spatial.cKDTree(mlonlat)
    
    # Cyclone path (converting to Numpy list)
    cyclone_pos = np.zeros((len(path),2))
    cyclone_pos[:,0] = path['lon']
    cyclone_pos[:,1] = path['lat']

    # Now, we create a loop that will find the data in new_bio that is within 50 km the cyclone path. 
    # The path dataset contains 39 positions of the cyclone throughout its activity, each with longitude and latitude. 
    # The loop will locate the  IDs (the index number) of the data in new_bio, within 50km of each of the 39 positions of the cyclone.
    # Creating the loop:
    for c in range(len(cyclone_pos)):
    
        ids = tree.query_ball_point(cyclone_pos[c,:], r=0.45)
        idlon = []
        idlat = []
        for k in range(len(ids)):
            ptlon = mlonlat[ids[k],0]
            ptlat = mlonlat[ids[k],1]
            idlon.append(np.where(ptlon == lon)[0][0])
            idlat.append(np.where(ptlat == lat)[0][0])
    
        # Saving the IDs for each position as a separate file (total 39 files, each file containing all the IDs of new_bio)
        data = {'idlon': idlon,
                'idlat': idlat}
        df = pd.DataFrame(data)
        df.to_csv('idclosetocyc'+str(year)+'/pos'+str(c)+'.csv',index=False)

    # List of variables to process
    variables = ['temp', 'Secchi', 'Chl_a_sum']
    
    # Create a dictionary to store statistics for each variable and position
    all_stats = {f'{var}_{pos}': {'Mean': [], 'Max': [], 'Min': []} for var in variables for pos in range(37) if not (25 <= pos <= 34)}
    
    # Loop through each position file, excluding: 
    # (1) pos0 - pos3 and pos37 to pos39 - because there is no data in these poisitons
    # (2) pos25 to pos34 - because the cyclone is above land in these poisitons 
    
    for pos in range(37):
        if pos in range(4) or 25 <= pos <= 34:
            continue
        
        # Read the corresponding CSV file
        idpos = pd.read_csv('idclosetocyc'+str(year)+'/pos'+str(pos)+'.csv')
        
        # Loop through each variable
        for var in variables:
            # Creating lists to store mean, max, and min values for all days in March 2016
            daily_means = []
            daily_maxs = []
            daily_mins = []
            
            # Loop through each day (time index from 0 to 30)
            for day in range(30):
                # Extract values for the specific day and position
                values = getattr(new_bio, var).isel(time=day).values[idpos['idlat'].values, idpos['idlon'].values]
                
                # Calculate statistics, ignoring NaNs
                mean_val = np.nanmean(values)
                max_val = np.nanmax(values)
                min_val = np.nanmin(values)
                
                # Append results to the lists
                daily_means.append(mean_val)
                daily_maxs.append(max_val)
                daily_mins.append(min_val)
            
            # Store the statistics for the current variable and position
            all_stats[f'{var}_{pos}']['Mean'] = daily_means
            all_stats[f'{var}_{pos}']['Max'] = daily_maxs
            all_stats[f'{var}_{pos}']['Min'] = daily_mins
    
    # Create a DataFrame combining mean, max, and min values for all variables
    data = {}
    for var in variables:
        for pos in range(37):
            if pos in range(4) or 25 <= pos <= 34:
                continue
            data[f'{var}_pos{pos}_Mean'] = all_stats[f'{var}_{pos}']['Mean']
            data[f'{var}_pos{pos}_Max'] = all_stats[f'{var}_{pos}']['Max']
            data[f'{var}_pos{pos}_Min'] = all_stats[f'{var}_{pos}']['Min']
    
    # Create DataFrame
    df_all_stats = pd.DataFrame(data, index=[f'Day {i+1}' for i in range(30)])
    
    # Transpose DataFrame to have days as columns and variables as rows
    df_all_stats_T = df_all_stats.T
    
    # Save the transposed DataFrame to a new CSV file
    df_all_stats_T.to_csv('combined_stats_March'+str(year)+'.csv')
    
    print('Combined statistics for temp, Secchi, and Chl_a_sum saved to combined_stats_March2017.csv')

    return






# Plotting the data 
# We create a function that will plot the data for each year:
def plot_data(listframe1, listframe2, listframe3, title):

    min_chl18 = listframe1[0]
    mean_chl18 = listframe1[1]
    max_chl18 = listframe1[2]

    min_se18 = listframe2[0]
    mean_se18 = listframe2[1]
    max_se18 = listframe2[2]

    min_t18 = listframe3[0]
    mean_t18 = listframe3[1]
    max_t18 = listframe3[2]
    
    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plotting the trendlines of Chlorophyll-a
    axes[0].plot(min_chl18[0], min_chl18[1], "r--", lw=2, label="Min Chlorophyll-a")
    axes[0].plot(mean_chl18[0], mean_chl18[1], "b--", lw=2, label="Mean Chlorophyll-a")
    axes[0].plot(max_chl18[0], max_chl18[1], "g--", lw=2, label="Max Chlorophyll-a")
    axes[0].set_title("Chlorophyll-a Trends "+title)
    axes[0].legend()
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Chlorophyll-a Anomaly")
    
    # Plotting the trendlines of Secchi
    axes[1].plot(min_se18[0], min_se18[1], "r--", lw=2, label="Min Secchi")
    axes[1].plot(mean_se18[0], mean_se18[1], "b--", lw=2, label="Mean Secchi")
    axes[1].plot(max_se18[0], max_se18[1], "g--", lw=2, label="Max Secchi")
    axes[1].set_title("Secchi Trends "+title)
    axes[1].legend()
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("Secchi Anomaly")
    
    # Plotting the trendlines of temperature
    axes[2].plot(min_t18[0], min_t18[1], "r--", lw=2, label="Min Temp")
    axes[2].plot(mean_t18[0], mean_t18[1], "b--", lw=2, label="Mean Temp")
    axes[2].plot(max_t18[0], max_t18[1], "g--", lw=2, label="Max Temp")
    axes[2].set_title("Temperature Trends "+title)
    axes[2].legend()
    axes[2].set_xlabel("Days")
    axes[2].set_ylabel("Temperature Anomaly")
    
    # Adjust the layout so the plots don't overlap
    plt.tight_layout()
    
    # Display the figure
    plt.show()

    return

# Creating a function to generate all the dataframes for each variable & Stat

def get_indices(lst, targets):
    return list(filter(lambda x: lst[x] in targets, range(len(lst))))
    
def getVarStat(new_norm, vname, stat):
    '''
    Extract from the dataframe the required values
    '''
    
    idx = get_indices(new_norm['var_name'], [vname])
    tmpdf = new_norm.iloc[idx].reset_index()
    del tmpdf['index']
    tmpdf
    
    idx = get_indices(tmpdf['stats'], [stat])
    tmpStat = tmpdf.iloc[idx].reset_index()
    del tmpStat['index']
    
    return tmpStat

# Creating a function to generate the trendlines of the anomalies 

def getTrend(df):
    tmp = []
    x = []
    for pos in range(len(df)):
        tmpval = df.iloc[pos].values[1:30]
        x = np.arange(len(tmpval))
        tmp.append(tmpval)
    xscatter = np.asarray(x, dtype=float).flatten()
    yscatter = np.asarray(tmp, dtype=float).flatten()
    ind = np.argsort(xscatter)
    fit = np.polyfit(xscatter[ind], yscatter[ind], deg=20) 
    p = np.poly1d(fit) 

    return [xscatter[ind],p(xscatter[ind])]

# Creating a function to generate the scatter plots & trendlines of the anomalies 

def plotVarStat(df):
    tmp = []
    x = []
    for pos in range(len(df)):
        tmpval = df.iloc[pos].values[1:30]
        x = np.arange(len(tmpval))#
        tmp.append(tmpval)
        plt.scatter(x, tmpval)
    xscatter = np.asarray(x, dtype=float).flatten()
    yscatter = np.asarray(tmp, dtype=float).flatten()
    ind = np.argsort(xscatter)
    fit = np.polyfit(xscatter[ind], yscatter[ind], deg=20) 
    p = np.poly1d(fit) 
    plt.plot(xscatter[ind],p(xscatter[ind]),"r--", lw=2) 
    plt.show()

    return [xscatter[ind],p(xscatter[ind])]

