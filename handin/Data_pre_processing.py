#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this script pre-processing of all data is performed. This involves:
downsampling, change of resolution and normalization
Finally new netCDF files are created
"""


# -- File info --#
__author__ = 'Ida L. Olsen'
__contributors__ = ''
__contact__ = ['s174020@student.dtu.dk']
__version__ = '0'
__date__ = '2021-10-01'

# -- Built-in modules -- #
import os.path

# -- Third-part modules -- #
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.signal import convolve2d

# -- Proprietary modules -- #
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

# load data
for file in os.listdir(directory):
    print(file)
    save_directory = '/work3/s174020/data/datasplit/CNN_data/'
    savefile = os.path.join(save_directory, file+'done')
    file_name_check = os.path.join(save_directory, savefile)
    if os.path.exists(file_name_check):
        print('file already exists')
        os.remove(file_name_check)
    # else:
    if file.endswith('_sub.nc'):
    # try:
        count += 1
        print(count)

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

        # Get ice concentrations
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

        # get distances to land and land fractions

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

        # plotting of land and SIC layers
        print('plotting')
        # import seaborn as sns
        from polar_plots import plot as pp
        #fig, ax = plt.subplots()
        #sns.heatmap(means)
        ID = file.replace('_S1A_AMSR2_Icechart', '')
        ID = ID.replace('_sub.nc', '')
        clim = [0, 1]
        extent = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)]
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

        ID = file.replace('_S1A_AMSR2_Icechart', '')
        ID = ID.replace('_sub.nc', '')
        clim = [0, 1]
        
        plt.figure(figsize=(9, 6))
        pp(lat, lon, mean_frac_land, extent, clim)
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

        # data normalization

        frequenciesH = ['6.9GHzH', '7.3GHzH', '10.7GHzH',
                        '18.7GHzH', '23.8GHzH', '36.5GHzH',
                        '89.0GHzH', '89.0aGHzH', '89.0bGHzH']

        frequenciesV = ['6.9GHzV', '7.3GHzV', '10.7GHzV',
                        '18.7GHzV', '23.8GHzV', '36.5GHzV',
                        '89.0GHzV', '89.0aGHzV', '89.0bGHzV']

        norm_tb69h = normalize_array(btemp69h, frequenciesH[0])
        norm_tb69v = normalize_array(btemp69v, frequenciesV[0])

        norm_tb73h = normalize_array(btemp73h, frequenciesH[1])
        norm_tb73v = normalize_array(btemp73v, frequenciesV[1])

        norm_tb107h = normalize_array(btemp107h, frequenciesH[2])
        norm_tb107v = normalize_array(btemp107v, frequenciesV[2])

        norm_tb187h = normalize_array(btemp187h, frequenciesH[3])
        norm_tb187v = normalize_array(btemp187v, frequenciesV[3])

        norm_tb238h = normalize_array(btemp238h, frequenciesH[4])
        norm_tb238v = normalize_array(btemp238v, frequenciesV[4])

        norm_tb365h = normalize_array(btemp365h, frequenciesH[5])
        norm_tb365v = normalize_array(btemp365v, frequenciesV[5])

        norm_tb890h = normalize_array(btemp890h, frequenciesH[6])
        norm_tb890v = normalize_array(btemp890v, frequenciesV[6])

        # normalize distances
        from normalize_arr_dist import normalize_array_dist
        norm_frac_land = normalize_array_dist(mean_frac_land, 'frac')
        norm_dist = normalize_array_dist(mean_distance, 'dist')

        #%% create .nc files

        save_directory = '/work3/s174020/data/datasplit/CNN_data/'
        savefile = os.path.join(save_directory, file+'done')
        # try:
        #     os.remove(savefile)
        # except:
        #     print('no file')
        f = nc4.Dataset(savefile, 'w',  format='NETCDF4')  # 'w' stands for write

        varname = ['norm_btemp_6.9v', 'norm_btemp_6.9h', 'norm_btemp_7.3v', 'norm_btemp_7.3h',
                   'norm_btemp_10.7v', 'norm_btemp_10.7h', 'norm_btemp_18.7v', 'norm_btemp_18.7h',
                   'norm_btemp_23.8v', 'norm_btemp_23.8h', 'norm_btemp_36.5v', 'norm_btemp_36.5h',
                   'norm_btemp_89.0v', 'norm_btemp_89.0h', 'norm_frac_land', 'norm_dist_land',
                   'norm_SIC_means', 'norm_SIC_modes']

        vardata = [norm_tb69v, norm_tb69h, norm_tb73v, norm_tb73h, norm_tb107v,
                   norm_tb107h, norm_tb187v, norm_tb187h, norm_tb238v,
                   norm_tb238h, norm_tb365v, norm_tb365h, norm_tb890v,
                   norm_tb890h, norm_frac_land, norm_dist, means, modes]

        f.createDimension('lon', np.shape(norm_tb69v)[0])
        f.createDimension('lat', np.shape(norm_tb69v)[1])

        norm_tb69v_cnn = f.createVariable(varname[0], 'f4', ('lon', 'lat'))
        norm_tb69h_cnn = f.createVariable(varname[1], 'f4', ('lon', 'lat'))
        norm_tb73v_cnn = f.createVariable(varname[2], 'f4', ('lon', 'lat'))
        norm_tb73h_cnn = f.createVariable(varname[3], 'f4', ('lon', 'lat'))
        norm_tb107v_cnn = f.createVariable(varname[4], 'f4', ('lon', 'lat'))
        norm_tb107h_cnn = f.createVariable(varname[5], 'f4', ('lon', 'lat'))
        norm_tb187v_cnn = f.createVariable(varname[6], 'f4', ('lon', 'lat'))
        norm_tb187h_cnn = f.createVariable(varname[7], 'f4', ('lon', 'lat'))
        norm_tb238v_cnn = f.createVariable(varname[8], 'f4', ('lon', 'lat'))
        norm_tb238h_cnn = f.createVariable(varname[9], 'f4', ('lon', 'lat'))
        norm_tb365v_cnn = f.createVariable(varname[10], 'f4', ('lon', 'lat'))
        norm_tb365h_cnn = f.createVariable(varname[11], 'f4', ('lon', 'lat'))
        norm_tb890v_cnn = f.createVariable(varname[12], 'f4', ('lon', 'lat'))
        norm_tb890h_cnn = f.createVariable(varname[13], 'f4', ('lon', 'lat'))
        norm_frac_land_cnn = f.createVariable(varname[14], 'f4', ('lon', 'lat'))
        norm_dist_cnn = f.createVariable(varname[15], 'f4', ('lon', 'lat'))
        SIC_means_cnn = f.createVariable(varname[16], 'f4', ('lon', 'lat'))
        SIC_modes_cnn = f.createVariable(varname[17], 'f4', ('lon', 'lat'))
        lat_cnn = f.createVariable('latitude', 'f4', ('lon', 'lat'))
        lon_cnn = f.createVariable('longitude', 'f4', ('lon', 'lat'))

        norm_tb69v_cnn[:, :] = norm_tb69v
        norm_tb69h_cnn[:, :] = norm_tb69h
        norm_tb73v_cnn[:, :] = norm_tb73v
        norm_tb73h_cnn[:, :] = norm_tb73h
        norm_tb107v_cnn[:, :] = norm_tb107v
        norm_tb107h_cnn[:, :] = norm_tb107h
        norm_tb187v_cnn[:, :] = norm_tb187v
        norm_tb187h_cnn[:, :] = norm_tb187h
        norm_tb238v_cnn[:, :] = norm_tb238v
        norm_tb238h_cnn[:, :] = norm_tb238h
        norm_tb365v_cnn[:, :] = norm_tb365v
        norm_tb365h_cnn[:, :] = norm_tb365h
        norm_tb890v_cnn[:, :] = norm_tb890v
        norm_tb890h_cnn[:, :] = norm_tb890h
        norm_frac_land_cnn[:, :] = norm_frac_land
        norm_dist_cnn[:, :] = norm_dist
        SIC_means_cnn[:, :] = means
        SIC_modes_cnn[:, :] = modes
        lat_cnn[:, :] = lat
        lon_cnn[:, :] = lon

        f.close()
