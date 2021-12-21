#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:20:05 2021

@author: s174020

This script extracts relevant data collumns from total dataset
"""

import xarray as xr
import os
from zipfile import BadZipfile
from zipfile import ZipFile

names_rel = ['polygon_codes',
             'lat',
             'lon',
             'sample',
             'line',
             'delays',
             'btemp_6.9h',
             'btemp_6.9v',
             'btemp_7.3h',
             'btemp_7.3v',
             'btemp_10.7h',
             'btemp_10.7v',
             'btemp_18.7h',
             'btemp_18.7v',
             'btemp_23.8h',
             'btemp_23.8v',
             'btemp_36.5h',
             'btemp_36.5v',
             'btemp_89.0ah',
             'btemp_89.0av',
             'btemp_89.0bh',
             'btemp_89.0bv',
             'btemp_89.0h',
             'btemp_89.0v',
             'polygon_icechart',
             'distance_map']


# Create a ZipFile Object and load sample.zip in it
count = 0
with ZipFile('/work3/s174020/data/13011134.zip', 'r') as zipObj:
    # Get a list of all archived file names from the zip
    listOfFileNames = zipObj.namelist()
    # Iterate over the file names
    for fileName in listOfFileNames:
        count += 1
        num_rel = len([i+1 for i in range(len(listOfFileNames)) if listOfFileNames[i].endswith('.nc')])
        # Check filename endswith .nc
        # define directory and save_path
        directory = '/work3/s174020/data/'
        save_path = directory+'datasplit'
        file_name_check = os.path.join(save_path, fileName[0:-3]) + '_sub.nc'
        if os.path.exists(file_name_check):
            print(count)
        else:
            if fileName.endswith('.nc'):
                
                # Extract a single file from zip
                try:
                    # keep track of number of files
                    print(count)
    
                    zipObj.extract(fileName)
                    # print(fileName)
    
                    # define file using complete path
                    fn = os.path.join(directory, fileName)
    
                    # remove file if it already exists (in case of changes)
                    # if os.path.join(save_path,fileName[0:-3])+'_sub.nc':
                    # if file already exist in folder when do not remove it! just move on
                    # try:
                    #     os.remove(file_name_check)
                    # except:
                    #     print('file not there')
    
                    # load data and make file wiht less variables
                    DS_all = xr.open_dataset(fn)
                    DS_sel = DS_all[names_rel]
    
                    try:
                        DS_sel.to_netcdf(os.path.join(save_path, fileName[0:-3])+'_sub.nc')
                        # print('saved file')
                    except:
                        print('file is already loaded, change name if you want new content or use code in the top')
                    # remove file
                    os.remove(os.path.join(directory, fileName))
                    # print('removed '+fileName)
                except BadZipfile:
                    # print('badly zipped file')
                    print(fileName)
                    break