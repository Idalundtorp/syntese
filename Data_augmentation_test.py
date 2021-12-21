#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Data augmentation script"""

# -- File info --#
__author__ = 'Ida L. Olsen'
__contributors__ = ''
__contact__ = ['s174020@student.dtu.dk']
__version__ = '0'
__date__ = '2021-11-12'

# -- Built-in modules -- #
import os

# -- Third-part modules -- #
import imgaug.augmenters as iaa
import netCDF4 as nc4
import numpy as np

# -- Proprietary modules -- #
from get_nc_CNN_data import read_final_data

# get data
directory = '/work3/s174020/data/datasplit/CNN_data'
count = 0
for file in os.listdir(directory):
    if file.endswith('.ncdone'):
        print(file)
        data = read_final_data(directory,  file)
        labels = data[17]  # SIC modes
        uni = np.unique(labels[~np.isnan(labels)])
        # weighted augmentation
        if uni.all() != 0:
            print(uni)
            if isinstance(data, list):
                seq = iaa.Sequential([iaa.Flipud(1)])  # flip all images vertically
                # seq = iaa.Sequential([iaa.Fliplr(1)])  # flip all images horizontally
    
                datadone = data
                count = 0
                for channel in data:
                    for i in range(len(channel)):
                        datadone[count] = seq.augment_images(channel)
                    count += 1
                    # fig  =  plt.subplots()
                    # sns.heatmap(datadone[0][0])
                    # fig  =  plt.subplots()
                    # sns.heatmap(datadone[0][99])
    
                save_directory = '/work3/s174020/data/datasplit/CNN_data/augmented_weighted'
                savefile = os.path.join(save_directory,  file + 'flip')
                try:
                    os.remove(savefile)
                except:
                    print('no file')
                f = nc4.Dataset(savefile, 'w', format='NETCDF4')  # 'w' stands for write
    
                varname = ['norm_btemp_6.9v', 'norm_btemp_6.9h', 'norm_btemp_7.3v', 'norm_btemp_7.3h',
                           'norm_btemp_10.7v', 'norm_btemp_10.7h', 'norm_btemp_18.7v', 'norm_btemp_18.7h',
                           'norm_btemp_23.8v', 'norm_btemp_23.8h', 'norm_btemp_36.5v', 'norm_btemp_36.5h',
                           'norm_btemp_89.0v', 'norm_btemp_89.0h', 'norm_frac_land', 'norm_dist_land',
                           'norm_SIC_means', 'norm_SIC_modes', 'latitude', 'longitude']
                vardata = datadone
    
                f.createDimension('lon',  np.shape(datadone[0])[0])
                f.createDimension('lat',  np.shape(datadone[0])[1])
    
                norm_tb69v_cnn = f.createVariable(varname[0],  'f4',  ('lon', 'lat'))
                norm_tb69h_cnn = f.createVariable(varname[1],  'f4',  ('lon', 'lat'))
                norm_tb73v_cnn = f.createVariable(varname[2],  'f4',  ('lon', 'lat'))
                norm_tb73h_cnn = f.createVariable(varname[3],  'f4',  ('lon', 'lat'))
                norm_tb107v_cnn = f.createVariable(varname[4],  'f4',  ('lon', 'lat'))
                norm_tb107h_cnn = f.createVariable(varname[5],  'f4',  ('lon', 'lat'))
                norm_tb187v_cnn = f.createVariable(varname[6],  'f4',  ('lon', 'lat'))
                norm_tb187h_cnn = f.createVariable(varname[7],  'f4',  ('lon', 'lat'))
                norm_tb238v_cnn = f.createVariable(varname[8],  'f4',  ('lon', 'lat'))
                norm_tb238h_cnn = f.createVariable(varname[9],  'f4',  ('lon', 'lat'))
                norm_tb365v_cnn = f.createVariable(varname[10],  'f4',  ('lon', 'lat'))
                norm_tb365h_cnn = f.createVariable(varname[11],  'f4',  ('lon', 'lat'))
                norm_tb890v_cnn = f.createVariable(varname[12],  'f4',  ('lon', 'lat'))
                norm_tb890h_cnn = f.createVariable(varname[13],  'f4',  ('lon', 'lat'))
                norm_dist_cnn = f.createVariable(varname[15],  'f4',  ('lon', 'lat'))
                norm_frac_land_cnn = f.createVariable(varname[14],  'f4',  ('lon', 'lat'))
                SIC_means_cnn = f.createVariable(varname[16],  'f4',  ('lon', 'lat'))
                SIC_modes_cnn = f.createVariable(varname[17],  'f4',  ('lon', 'lat'))
                # lat_cnn = f.createVariable(varname[18],  'f4',  ('lon', 'lat'))
                # lon_cnn = f.createVariable(varname[19],  'f4',  ('lon', 'lat'))
    
                norm_tb69v_cnn[:, :] = datadone[0]
                norm_tb69h_cnn[:, :] = datadone[1]
                norm_tb73v_cnn[:, :] = datadone[2]
                norm_tb73h_cnn[:, :] = datadone[3]
                norm_tb107v_cnn[:, :] = datadone[4]
                norm_tb107h_cnn[:, :] = datadone[5]
                norm_tb187v_cnn[:, :] = datadone[6]
                norm_tb187h_cnn[:, :] = datadone[7]
                norm_tb238v_cnn[:, :] = datadone[8]
                norm_tb238h_cnn[:, :] = datadone[9]
                norm_tb365v_cnn[:, :] = datadone[10]
                norm_tb365h_cnn[:, :] = datadone[11]
                norm_tb890v_cnn[:, :] = datadone[12]
                norm_tb890h_cnn[:, :] = datadone[13]
                norm_dist_cnn[:, :] = datadone[14]
                norm_frac_land_cnn[:, :] = datadone[15]
                SIC_means_cnn[:, :] = datadone[16]
                SIC_modes_cnn[:, :] = datadone[17]
                # lat_cnn[:, :] = datadone[18]
                # lon_cnn[:, :] = datadone[19]
                
                f.close()