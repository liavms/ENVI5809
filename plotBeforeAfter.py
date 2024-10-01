# load the necessary packages 

import warnings
warnings.filterwarnings("ignore")
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




# Plotting the data 
# since the during period is not the same length as the before and after (during = 9 days, before & after = 22 days), we need to 
# adjust the code manually to get the data for the during period. 
# However, we create a function that will plot the data for the before and after period:


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
        tmpval = df.iloc[pos].values[0:21]
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
        tmpval = df.iloc[pos].values[0:21]
        x = np.arange(len(tmpval))
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
