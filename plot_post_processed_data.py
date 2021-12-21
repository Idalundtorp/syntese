#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:01:04 2021

@author: s174020
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:04:29 2021

@author: s174020

"""
import numpy as np
import os.path
import netCDF4 as nc4
import matplotlib.pyplot as plt
import xarray as xr
# import pandas as pd
# import xarray as xr
from scipy.signal import convolve2d
from normalize_arr import normalize_array
from torch.utils.data import DataLoader

from dataloader import ASIPDataset
from Get_weights import Get_weights
from Get_weights_org import Get_weights_org
from validation_dataloader import ASIPDatasetVal

directory = '/work3/s174020/data/datasplit/CNN_data/'
file = '20190427T211044_S1A_AMSR2_Icechart-Greenland-SouthWest_sub.ncdone'
# for file in os.listdir(directory):
#     if '.ncdone' in file:
#         print(file)

fn = os.path.join(directory, file)

ds4 = xr.open_dataset(fn)
# load TBs


lat = ds4['latitude'][:]
lon = ds4['longitude'][:]

btemp69v = ds4['norm_btemp_6.9v'][:]
btemp69h = ds4['norm_btemp_6.9h'][:]

btemp73v = ds4['norm_btemp_7.3v'][:]
btemp73h = ds4['norm_btemp_7.3h'][:]

btemp107v = ds4['norm_btemp_10.7v'][:]
btemp107h = ds4['norm_btemp_10.7h'][:]

btemp187v = ds4['norm_btemp_18.7v'][:]
btemp187h = ds4['norm_btemp_18.7h'][:]

btemp238v = ds4['norm_btemp_23.8v'][:]
btemp238h = ds4['norm_btemp_23.8h'][:]

btemp365v = ds4['norm_btemp_36.5v'][:]
btemp365h = ds4['norm_btemp_36.5h'][:]

btemp890v = ds4['norm_btemp_89.0v'][:]
btemp890h = ds4['norm_btemp_89.0h'][:]

frac_land = ds4['norm_frac_land'][:]
mean_distance = ds4['norm_dist_land'][:]

means = ds4['norm_SIC_means'][:]
modes = ds4['norm_SIC_modes'][:]

#frac_land[~np.isnan(frac_land)]
#%%
print('plotting')
# import seaborn as sns
from polar_plots import plot as pp
#fig, ax = plt.subplots()
#sns.heatmap(means)
ID = file.replace('_S1A_AMSR2_Icechart', '')
ID = ID.replace('_sub.nc', '')
clim = [-1, 1]
extent = [np.nanmin(lon),np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
plt.figure(figsize=(9, 6))
pp(lat, lon, means, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Calculating of SIC using modes', fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

plt.figure(figsize=(9, 6))
pp(lat, lon, modes, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Calculating of SIC using modes', fontsize=14)
plt.show()

#%% 
print('plotting land')
ID = file.replace('_S1A_AMSR2_Icechart', '')
ID = ID.replace('_sub.nc', '')
clim = [-1, 1]

plt.figure(figsize=(9, 6))
pp(lat, lon, frac_land, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Fraction of land calculated', fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

clim = [-1, 1]
plt.figure(figsize=(9, 6))
pp(lat, lon, mean_distance, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Distance to land calculated', fontsize=14)
plt.show()


print('plotting AMSR2')
ID = file.replace('_S1A_AMSR2_Icechart', '')
ID = ID.replace('_sub.nc', '')
clim = [-1, 1]

clim = [np.min(btemp890v), np.max(btemp890v)]
plt.figure(figsize=(9, 6))
pp(lat, lon, btemp890v, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Fraction of land calculated', fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

clim = [np.min(btemp365v), np.max(btemp365v)]
plt.figure(figsize=(9, 6))
pp(lat, lon, btemp365v, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Distance to land calculated', fontsize=14)
plt.show()