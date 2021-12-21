#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:00:26 2021

@author: s174020
"""
#%% Default settings and packages
import numpy as np
import os.path
import nctoolkit as nc
import netCDF4 as nc4
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd


directory='/work3/s174020/data/datasplit'
#filename='20180327T104059_S1B_AMSR2_Icechart-Greenland-CentralWest_sub.nc'
#filename='20180413T202620_S1B_AMSR2_Icechart-Greenland-CapeFarewell_sub.nc'
filename='20180404T205505_S1B_AMSR2_Icechart-Greenland-CentralWest_sub.nc'
fn=os.path.join(directory,filename)
area='CentralWest'

##set matplotlib sizes
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#%%
DS_all = xr.open_dataset(fn)
DS_all

ds4 = nc4.Dataset(fn)

polygon_icechart = ds4['polygon_icechart'][:]
polygon_codes = ds4['polygon_codes'][:]

# #load TBs
lat = ds4['lat'][:]
lon = ds4['lon'][:]

# sar lat and lon
sarlat = ds4['sar_grid_latitude'][:]
sarlon = ds4['sar_grid_longitude'][:]

btemp69v = ds4['btemp_6.9v'][:]
btemp69h = ds4['btemp_6.9h'][:]

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

# btemp890av=ds4['btemp_89.0av'][:]
# btemp890ah=ds4['btemp_89.0ah'][:]

# btemp890bv=ds4['btemp_89.0bv'][:]
# btemp890bh=ds4['btemp_89.0bh'][:]

# %% Get ice concentrations
# print(polygon_codes)
# print(np.shape(polygon_codes))
# print(len(polygon_codes))

polygon_codes[1][0]
xm = polygon_icechart.compressed()

SIC = np.zeros((np.shape(polygon_icechart)))
# CT=00 -> SIC=0

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



# %%Plot histograms of data
# from plot_histograms import plot_histograms as ph
# ph(fn)

# %% Show area
from polar_plots import plot

# hvor er AMSR lat og lon filer og tidspunkter?
plt.rcParams['figure.figsize'] = [3, 5]  # number of inches of figure (width,height)
extent = [np.min(lon)-3, np.max(lon)+3, np.min(lat)-0.5, np.max(lat)+0.5]

area = area

fig, ax = plt.subplots()
plot(lat, lon, btemp69v, extent, [180, 300])
plt.title('6.9GHzV', loc='left')
plt.savefig('69GHzV'+area+'.jpg')
# plt.close()

# %% Calculate means of segments
# from scipy import stats
# #stats.mode(data)
# reshape array
rowlen = len(SIC[0, :])
collen = len(SIC[:, 0])

rows_AMSR2 = int((rowlen-(rowlen % 50))/50)
cols_AMSR2 = int((collen-(collen % 50))/50)

arr_in_pol_ice = np.reshape(SIC[0:collen-(collen % 50),0:rowlen-(rowlen % 50)],(cols_AMSR2,50,rows_AMSR2,50))

means = np.ones(np.shape(lat))*np.nan
means[0:len(arr_in_pol_ice[:, 0, 0, 0]), 0:len(arr_in_pol_ice[0, 0, :, 0])] = np.nanmean(arr_in_pol_ice, axis=(1, 3))
# arr_left_row = np.mean(SIC[0,rowlen-(rowlen % 50):])
# arr_left_col = np.mean(SIC[collen-(collen % 50):,0])

# poscol = np.shape(lat)[0]-(np.shape(lat)[0]-len(arr_in_pol_ice[:,0,0,0]))
# posrow = np.shape(lat)[1]-(np.shape(lat)[1]-len(arr_in_pol_ice[0,0,:,0]))
# means[poscol,posrow] = [arr_left_col,arr_left_row]

# median
medians = np.ones(np.shape(lat))*np.nan
medians[0:len(arr_in_pol_ice[:, 0, 0, 0]), 0:len(arr_in_pol_ice[0, 0, :, 0])] = np.nanmedian(arr_in_pol_ice,axis = (1,3))


modes = np.ones(np.shape(lat))*np.nan

# colvalues = []
from faster_mode_1d import mode
[v, c] = mode(arr_in_pol_ice, axis=1)
[v1, c1] = mode(v, axis=2)

# [v,c] =  stats.mode(arr_in_pol_ice,axis = 1,nan_policy = 'omit')
# [v1,c1] =  stats.mode(arr_in_pol_ice,axis = 3,nan_policy = 'omit')
modes[0:len(arr_in_pol_ice[:, 0, 0, 0]), 0:len(arr_in_pol_ice[0, 0, :, 0])] = v1
# modes[0:len(arr_in_pol_ice[:,0,0,0]),0:len(arr_in_pol_ice[0,0,:,0])]  =  [colvalues,rowvalues]



from convert_dist import convert_dist as cd
dist = cd(ds4['distance_map'][:])  # km

rowlen = len(dist[0, :])
collen = len(dist[:, 0])

rows_AMSR2 = int((rowlen-(rowlen % 50))/50)
cols_AMSR2 = int((collen-(collen % 50))/50)

arr_in_pol_ice = np.reshape(dist[0:collen-(collen % 50), 0:rowlen-(rowlen % 50)], (cols_AMSR2, 50, rows_AMSR2, 50))

dist_means = np.ones(np.shape(lat))*np.nan
dist_means[0:len(arr_in_pol_ice[:, 0, 0, 0]), 0:len(arr_in_pol_ice[0, 0, :, 0])] = np.nanmean(arr_in_pol_ice,
                                                                                              axis=(1, 3))
# dist_means = cd(dist_means)


# dist = cd(ds4['distance_map'][:]) #km
dist = cd(ds4['distance_map'][:])
arr_in_pol_ice = np.reshape(dist[0:collen-(collen % 50), 0:rowlen-(rowlen % 50)], (cols_AMSR2, 50, rows_AMSR2, 50))

dist_medians = np.ones(np.shape(lat))*np.nan
dist_medians[0:len(arr_in_pol_ice[:, 0, 0, 0]), 0:len(arr_in_pol_ice[0, 0, :, 0])] = np.nanmedian(arr_in_pol_ice,
                                                                                                  axis=(1, 3))

# dist_medians = cd(dist_medians)
# arr_left_row = np.mean(dist[0,rowlen-(rowlen % 50):])
# arr_left_col = np.mean(dist[collen-(collen % 50):,0])

# poscol = np.shape(lat)[0]-(np.shape(lat)[0]-len(arr_in_pol_ice[:,0,0,0]))
# posrow = np.shape(lat)[1]-(np.shape(lat)[1]-len(arr_in_pol_ice[0,0,:,0]))
# means[poscol,posrow] = [arr_left_col,arr_left_row]
# %%scatterplots
fig, ax = plt.subplots()
plt.scatter(means, medians, s=70, facecolor=(0, 0.4470, 0.7410, .1), edgecolors=(0, 0.4470, 0.7410, .2))
plt.grid()
ax.set_aspect('equal', adjustable='box')
plt.xlim(-0.05, np.nanmax(means)+0.1*np.nanmax(means))
plt.ylim(-0.05, np.nanmax(means)+0.1*np.nanmax(means))
plt.xlabel('means')
plt.ylabel('medians')
plt.title('mean vs. median calculation of SIC')
plt.show()

fig,  ax = plt.subplots()
plt.scatter(medians,  modes,  s=100,  facecolor=(0,  0.4470,  0.7410,  .1),  edgecolors=(0,  0.4470,  0.7410,  1))
plt.grid()
ax.set_aspect('equal',  adjustable='box')
plt.xlim(-0.05, np.nanmax(medians)+0.1*np.nanmax(medians))
plt.ylim(-0.05, np.nanmax(medians)+0.1*np.nanmax(medians))
plt.xlabel('medians')
plt.ylabel('modes')
plt.title('median vs. mode calculation of SIC')

fig,  ax = plt.subplots()
plt.grid()
ax.set_aspect('equal', adjustable='box')
plt.xlim(-0.05, np.nanmax(means)+0.1*np.nanmax(means))
plt.ylim(-0.05, np.nanmax(means)+0.1*np.nanmax(means))
plt.scatter(means, modes, s=70, facecolor=(0,  0.4470,  0.7410, .1), edgecolors=(0,  0.4470,  0.7410, .2))
plt.xlabel('means')
plt.ylabel('modes')
plt.title('mean vs. mode calculation of SIC')

fig,  ax = plt.subplots()
plt.grid()
ax.set_aspect('equal', adjustable='box')
plt.xlim(-0.05, np.nanmax(dist_means)+0.1*np.nanmax(dist_means))
plt.ylim(-0.05, np.nanmax(dist_means)+0.1*np.nanmax(dist_means))

plt.scatter(dist_means, dist_medians, s=60, facecolor=(0,  0.4470,  0.7410, .1), edgecolors=(0,  0.4470,  0.7410, .2))
plt.xlabel('means')
plt.ylabel('medians')
plt.title('mean vs. median calculation of DIST')

fig, ax = plt.subplots()
Nr_pr_bin = 20
n, bins, patches = ax.hist(x=c1[~np.isnan(v1)].flatten(), bins=Nr_pr_bin, color='#0504aa',
                           alpha=0.7, rwidth=0.85, density=True)
ax.set_title('Histogram of counts per mode')
plt.xlabel('Number of elements that is mode')
plt.grid()
plt.ylabel('Density')

# %% make geographical plots
from polar_plots import plot

plt.rcParams['figure.figsize'] = [5,  7]  # number of inches of figure (width, height)
extent = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)]

area = area

fig,  ax = plt.subplots()
plot(lat, lon, means, extent, [0, 1])
plt.title('SIC-AMSR2-grid',  loc='left')
plt.savefig('SIC-AMSR2-grid'+area+'.jpg')

fig,  ax = plt.subplots()
plot(lat, lon, medians, extent, [0, 1])
plt.title('SIC-AMSR2-grid',  loc='left')
plt.savefig('SIC-AMSR2-grid'+area+'.jpg')

fig,  ax = plt.subplots()
plot(lat, lon, modes, extent, [0, 1])
plt.title('SIC-AMSR2-grid',  loc='left')
plt.savefig('SIC-AMSR2-grid'+area+'.jpg')

# #dist conversion
fig,  ax = plt.subplots()
plot(lat[dist_means > 0], lon[dist_means > 0], dist_means[dist_means > 0], extent, [0, 200])
plt.title('SIC-AMSR2-grid',  loc='left')
plt.savefig('dist-AMSR2-grid'+area+'.jpg')

fig,  ax = plt.subplots()
plot(lat[dist_medians > 0], lon[dist_medians > 0], dist_medians[dist_medians > 0], extent, [0, 200])
plt.title('SIC-AMSR2-grid',  loc='left')
plt.savefig('dist-AMSR2-grid'+area+'.jpg')
# %% Plot Sar image for comparison
# sar lat and lon
# sarlat=ds4['sar_grid_latitude'][:]
# sarlon=ds4['sar_grid_longitude'][:]
# sarsamps=ds4['sar_grid_sample'][:]
# sarlines=ds4['sar_grid_line'][:]
# sarhgts=ds4['sar_grid_height'][:]
# sarprimary=ds4['sar_primary'][:]


# from osgeo import gdal, ogr, osr
# import netCDF4, os, numpy
# #arr_to_put_in_gtiff = numpy.array(ncf.variables.get('sar_primary'))
# out_gtiff_filename = fn+'_sea_ice_predictions.tif'
# distmap=ds4['distance_map'][:]
# arr_to_put_in_gtiff = np.nan_to_num(distmap)
# rows, cols = arr_to_put_in_gtiff.shape

# spr_gcp = osr.SpatialReference()
# spr_gcp.ImportFromEPSG(4326)

# gcps=()

# for i in range(len(sarlines)):
#     x, y, z, pix, lin = sarlon[i],sarlat[i],sarhgts[i],sarsamps[i],sarlines[i]
#     gcps = gcps + (gdal.GCP(x, y, z, pix, lin, '', str(i)),)

# #AI4Arctic Sea Ice Dataset -User Manual36
# # NOTE: gdal.GDT_Float32 must be changed to gdal.GDT_"something_else" if output isanother type
# gtiff = gdal.GetDriverByName('GTiff').Create(out_gtiff_filename, cols, rows, 1, gdal.GDT_Float32) 
# gtiff.GetRasterBand(1).WriteArray(arr_to_put_in_gtiff)
# gtiff.SetGCPs(gcps, spr_gcp.ExportToWkt())
# gtiff = None

# import rasterio
# from rasterio.plot import show
# fp = fn+'_sea_ice_predictions.tif'

# img = rasterio.open(fp)
# fig, ax = plt.subplots(figsize=(10,10))
# image_hidden = ax.imshow(img.read()[0])
# fig.colorbar(image_hidden, ax=ax)
# rasterio.plot.show(img, ax=ax)
