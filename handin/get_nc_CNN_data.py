#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:56:23 2021

@author: s174020
"""

import numpy as np
import os.path
import netCDF4 as nc4
import matplotlib.pyplot as plt

def read_final_data(directory, file):

    # dist_means_all = []

    # norm_tb69h_all = []
    # norm_tb69v_all = []
    # norm_tb73h_all = []
    # norm_tb73v_all = []
    # norm_tb107h_all = []
    # norm_tb107v_all = []
    # norm_tb187h_all = []
    # norm_tb187v_all = []
    # norm_tb238h_all = []
    # norm_tb238v_all = []
    # norm_tb365h_all = []
    # norm_tb365v_all = []
    # norm_tb890h_all = []
    # norm_tb890v_all = []

    # norm_dist_all = []
    # norm_frac_land_all = []
    # SIC_means_all = []
    # SIC_modes_all = []


    try:
        fn = os.path.join(directory, file)
        ds4 = nc4.Dataset(fn)
        norm_tb69v = ds4['norm_btemp_6.9v'][:]
        norm_tb69h = ds4['norm_btemp_6.9h'][:]
    
        norm_tb73v = ds4['norm_btemp_7.3v'][:]
        norm_tb73h = ds4['norm_btemp_7.3h'][:]
    
        norm_tb107v = ds4['norm_btemp_10.7v'][:]
        norm_tb107h = ds4['norm_btemp_10.7h'][:]
    
        norm_tb187v = ds4['norm_btemp_18.7v'][:]
        norm_tb187h = ds4['norm_btemp_18.7h'][:]
    
        norm_tb238v = ds4['norm_btemp_23.8v'][:]
        norm_tb238h = ds4['norm_btemp_23.8h'][:]
    
        norm_tb365v = ds4['norm_btemp_36.5v'][:]
        norm_tb365h = ds4['norm_btemp_36.5h'][:]
    
        norm_tb890v = ds4['norm_btemp_89.0v'][:]
        norm_tb890h = ds4['norm_btemp_89.0h'][:]
    
        norm_frac_land = ds4['norm_frac_land'][:]
        norm_dist = ds4['norm_dist_land'][:]
    
        SIC_means = ds4['norm_SIC_means'][:]
        SIC_modes = ds4['norm_SIC_modes'][:]
        
        return [norm_tb69v,norm_tb69h,norm_tb73v,norm_tb73h,norm_tb107v,
                                 norm_tb107h,norm_tb187v,norm_tb187h,norm_tb238v,
                                 norm_tb238h,norm_tb365v,norm_tb365h,norm_tb890v,
                                 norm_tb890h,norm_frac_land,norm_dist,SIC_means,SIC_modes]


    except:
        print('error - no data in file')


    #         norm_tb69h_all.append(norm_tb69h)
    #         norm_tb69v_all.append(norm_tb69v)
    #         norm_tb73h_all.append(norm_tb73h)
    #         norm_tb73v_all.append(norm_tb73h)
    #         norm_tb107h_all.append(norm_tb107h)
    #         norm_tb107v_all.append(norm_tb107v)
    #         norm_tb187h_all.append(norm_tb187h)
    #         norm_tb187v_all.append(norm_tb187v)
    #         norm_tb238h_all.append(norm_tb238h)
    #         norm_tb238v_all.append(norm_tb238v)
    #         norm_tb365h_all.append(norm_tb365h)
    #         norm_tb365v_all.append(norm_tb365v)
    #         norm_tb890h_all.append(norm_tb890h)
    #         norm_tb890v_all.append(norm_tb890v)
    #         norm_frac_land_all.append(norm_frac_land)
    #         norm_dist_all.append(norm_dist)
    #         SIC_means_all.append(SIC_means)
    #         SIC_modes_all.append(SIC_modes)
            
    # return [norm_tb69v_all, norm_tb69h_all,
    #         norm_tb73v_all, norm_tb73h_all,
    #         norm_tb107v_all, norm_tb107h_all,
    #         norm_tb187v_all, norm_tb187h_all,
    #         norm_tb238v_all, norm_tb238h_all,
    #         norm_tb365v_all, norm_tb365h_all,
    #         norm_tb890v_all, norm_tb890h_all,
    #         norm_frac_land_all, norm_dist_all,
    #         SIC_means_all, SIC_modes_all]
