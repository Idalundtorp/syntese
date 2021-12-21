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


directory = '/work3/s174020/data/datasplit'
count = 0

norm_tb69h_all = []
norm_tb69v_all = []
norm_tb73h_all = []
norm_tb73v_all = []
norm_tb107h_all = []
norm_tb107v_all = []
norm_tb187h_all = []
norm_tb187v_all = []
norm_tb238h_all = []
norm_tb238v_all = []
norm_tb365h_all = []
norm_tb365v_all = []
norm_tb890h_all = []
norm_tb890v_all = []

norm_dist_all = []
norm_frac_land_all = []
SIC_means = []
SIC_modes = []

directory = '/work3/s174020/data/datasplit'
file = '20180314T202722_S1A_AMSR2_Icechart-Greenland-CapeFarewell_sub.nc'
# file = '20190104T103202_S1B_AMSR2_Icechart-Greenland-CentralWest_sub.nc'
file = '20190427T211044_S1A_AMSR2_Icechart-Greenland-SouthWest_sub.nc'

fn = os.path.join(directory, file)

ds4 = xr.open_dataset(fn)
# load TBs
lat = ds4['lat'][:]
lon = ds4['lon'][:]

btemp69v = ds4['btemp_6.9v'][:]
btemp69h = ds4['btemp_6.9h'][:]

btemp73v = ds4['btemp_7.3v'][:]
btemp73h = ds4['btemp_7.3h'][:]

btemp107v = ds4['btemp_10.7v'][:]
btemp107h = ds4['btemp_10.7h'][:]

btemp187v = ds4['btemp_18.7v'][:]
btemp187h = ds4['btemp_18.7h'][:]

btemp238v = ds4['btemp_23.8v'][:]
btemp238h = ds4['btemp_23.8h'][:]

btemp365v = ds4['btemp_36.5v'][:]
btemp365h = ds4['btemp_36.5h'][:]

btemp890v = ds4['btemp_89.0v'][:]
btemp890h = ds4['btemp_89.0h'][:]

ds4_2 = nc4.Dataset(fn)
polygon_icechart = ds4_2['polygon_icechart'][:]
polygon_codes = ds4_2['polygon_codes'][:]


#%% Get ice concentrations
polygon_codes[1][0]
xm = polygon_icechart.compressed()

SIC = np.zeros((np.shape(polygon_icechart)))

for row in polygon_codes:
    row = row.split(";")
    if row[0] != 'id':
        id = int(row[0])
        SIC[polygon_icechart == id] = int(row[1])

SIC[SIC == 0] = np.nan
SIC[SIC == 1] = 0.5/10  # less that 10%
SIC[SIC == 2] = 0/10  # ocean water
SIC[SIC == 10] = 1/10
SIC[SIC == 20] = 2/10
SIC[SIC == 30] = 3/10
SIC[SIC == 40] = 4/10
SIC[SIC == 50] = 5/10
SIC[SIC == 60] = 6/10
SIC[SIC == 70] = 7/10
SIC[SIC == 80] = 8/10
SIC[SIC == 90] = 9/10
SIC[SIC == 91] = 9.5/10
SIC[SIC == 92] = 10/10

# Calculate means of segments
rowlen = len(SIC[0, :])
collen = len(SIC[:, 0])

rows_AMSR2 = int((rowlen-(rowlen % 50))/50)
cols_AMSR2 = int((collen-(collen % 50))/50)

arr_in_pol_ice = np.reshape(SIC[0:collen-(collen % 50), 0:rowlen-(rowlen % 50)],
                            (cols_AMSR2, 50, rows_AMSR2, 50))

means = np.ones(np.shape(lat))*np.nan
means[0:len(arr_in_pol_ice[:, 0, 0, 0]),
      0:len(arr_in_pol_ice[0, 0, :, 0])] = np.nanmean(arr_in_pol_ice, axis=(1, 3))

modes = np.ones(np.shape(lat))*np.nan

from faster_mode_1d import mode
[v, c] = mode(arr_in_pol_ice, axis=1)
[v1, c1] = mode(v, axis=2)

modes[0:len(arr_in_pol_ice[:, 0, 0, 0]), 0:len(arr_in_pol_ice[0, 0, :, 0])] = v1

SIC_means = np.append(SIC_means, means)

SIC_modes = np.append(SIC_modes, modes)

#%%
print('plotting')
extent = [np.nanmin(lon),np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
# import seaborn as sns
from polar_plots_pred import plot as pp
#fig, ax = plt.subplots()
#sns.heatmap(means)
ID = file.replace('_S1A_AMSR2_Icechart', '')
ID = ID.replace('_sub.nc', '')
clim = [0, 1]
#extent = [-52, -35, 56, 63]
plt.figure(figsize=(9, 6))
pp(lat, lon, means, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Calculating of SIC using means', fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

plt.figure(figsize=(9, 6))
pp(lat, lon, modes, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Calculating of SIC using modes', fontsize=14)
plt.show()

#%% get distances and fractions

from convert_dist import convert_dist as cd
dist = cd(ds4['distance_map'][:])  # km

rowlen = len(dist[0, :])
collen = len(dist[:, 0])

rows_AMSR2 = int((rowlen-(rowlen % 50))/50)
cols_AMSR2 = int((collen-(collen % 50))/50)

arr_in_pol_ice = np.reshape(dist[0:collen-(collen % 50), 0:rowlen-(rowlen % 50)],
                            (cols_AMSR2, 50, rows_AMSR2, 50))

# find land areas
landmask = (arr_in_pol_ice == 0)
frac_land = np.ones(np.shape(lat))*np.nan

for i in range(len(landmask[:, 0, 0, 0])):
    for j in range(len(landmask[0, 0, :, 0])):
        if np.any(landmask[i, :, j, :]):
            counts = np.where(landmask[i, :, j, :].flatten() == True)
            frac_land[i, j] = len(counts[0]) / (50*50)
        else:
            frac_land[i, j] = 0

dist_means = np.ones(np.shape(lat))*np.nan
dist_means[0:len(arr_in_pol_ice[:, 0, 0, 0]),
           0:len(arr_in_pol_ice[0, 0, :, 0])] = np.nanmean(arr_in_pol_ice, axis=(1, 3))

w = 7
h = 7
kern = np.ones((w, h))*1 / (w*h)

mean_distance = convolve2d(dist_means, kern, boundary='symm', mode='same')
mean_frac_land = convolve2d(frac_land, kern, boundary='symm', mode='same')

#%% 
print('plotting land')
ID = file.replace('_S1A_AMSR2_Icechart', '')
ID = ID.replace('_sub.nc', '')
clim = [0, 1]

#extent = [np.nanmin(lon),np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
#extent = [-52, -35, 56, 63]
plt.figure(figsize=(9, 6))
pp(lat, lon, frac_land, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Fraction of land calculated', fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

clim = [0, 300]
plt.figure(figsize=(9, 6))
pp(lat, lon, mean_distance, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('Distance to land calculated', fontsize=14)
plt.show()


print('plotting AMSR2')
ID = file.replace('_S1A_AMSR2_Icechart', '')
ID = ID.replace('_sub.nc', '')
clim = [0, 1]

#extent = [-52, -35, 56, 63]
clim = [np.min(btemp890v), np.max(btemp890v)]
plt.figure(figsize=(9, 6))
pp(lat, lon, btemp890v, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('TB 89.0 GHz V', fontsize=14)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

clim = [np.min(btemp365v), np.max(btemp365v)]
plt.figure(figsize=(9, 6))
pp(lat, lon, btemp365v, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('TB 36.5 GHz V', fontsize=14)
plt.show()

clim = [np.min(btemp187v), np.max(btemp187v)]
plt.figure(figsize=(9, 6))
pp(lat, lon, btemp365v, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('TB 18.7 GHz V', fontsize=14)
plt.show()

clim = [np.min(btemp69v), np.max(btemp69v)]
plt.figure(figsize=(9, 6))
pp(lat, lon, btemp69v, extent, clim)
plt.suptitle(ID, fontsize=16)
plt.title('TB 6.9 GHz V', fontsize=14)
plt.show()
