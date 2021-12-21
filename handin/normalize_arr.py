#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:18:08 2021

@author: s174020
 Function for normalization
"""

import numpy as np

def normalize_array(an_array,frequency):
    frequencies=['6.9GHzV','6.9GHzH','7.3GHzV','7.3GHzH','10.7GHzV','10.7GHzH'
                 ,'18.7GHzV','18.7GHzH','23.8GHzV','23.8GHzH','36.5GHzV','36.5GHzH'
                 ,'89.0GHzV','89.0GHzH','89.0aGHzV','89.0aGHzH','89.0bGHzV','89.0bGHzH']
    #frequency='6.9GHzV'
    #an_array=btemp69v
    
    if frequency in frequencies:
        i = 0
        length = len(frequencies)
        
        while i < length:
            if frequency == frequencies[i]:
                ind=int(i)
                break
            i += 1
            
        min_values = np.load('norm_vals.npy')
        max_values = np.load('norm_vals_max.npy')
        
        b=1
        a=-1
        normal_arr=a+((an_array-min_values[ind])*(b-a))/(max_values[ind]-min_values[ind])
    else:
        print('invalid frequency entered')
    
    return normal_arr
