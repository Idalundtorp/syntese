#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:08:55 2021

@author: s174020
"""

from sklearn.utils import class_weight
import numpy as np
import os
import xarray as xr
import torch


def Get_weights_org(directory):

    train_labels = []
    train_data = []
    count = 0
    # directory = '/work3/s174020/data/datasplit/CNN_data/'
    val_scenes = np.genfromtxt(directory + 'validation_scenes.txt', dtype=str,
                               skip_header=2)
    invalid_scenes = np.genfromtxt(directory + 'invalid_scenes.txt', dtype=str,
                                   skip_header=2)
    for file in os.listdir(directory):
        if '.ncdone' in file:
            fn = os.path.join(directory, file)
            ds = xr.open_dataset(fn)
            ndarr = ds.to_array()
            labels = ndarr[17, :, :].values  # SIC modes
            # data = ndarr[:16, :, :].values   # all other channels expect SIC means
            size = (256, 256)
            shape = np.shape(labels)
    
            pad_size = (size[0]-shape[0], size[1]-shape[1])
            # data[np.isnan(data)] = 0
            # labels[labels == 10/10] = 7
            # labels[labels == 9.5/10] = 7
            # labels[labels == 9/10] = 6
            # labels[labels == 8/10] = 5
            # labels[labels == 7/10] = 5
            # labels[labels == 6/10] = 4
            # labels[labels == 5/10] = 4
            # labels[labels == 4/10] = 3
            # labels[labels == 3/10] = 3
            # labels[labels == 2/10] = 2
            # labels[labels == 1/10] = 2
            # labels[labels == 0.5/10] = 1  # less that 10%
            # labels[labels == 0/10] = 1  # ocean water
            # labels[np.isnan(labels)] = 0
            # labels[labels < 0] = 0
            # labels[labels > 11] = 0
            
            labels[labels == 10/10] = 11
            labels[labels == 9.5/10] = 11
            labels[labels == 9/10] = 10
            labels[labels == 8/10] = 9
            labels[labels == 7/10] = 8
            labels[labels == 6/10] = 7
            labels[labels == 5/10] = 6
            labels[labels == 4/10] = 5
            labels[labels == 3/10] = 4
            labels[labels == 2/10] = 3
            labels[labels == 1/10] = 2
            labels[labels == 0.5/10] = 1  # less that 10%
            labels[labels == 0/10] = 1  # ocean water
            labels[np.isnan(labels)] = 0
            labels[labels < 0] = 0
            labels[labels > 11] = 0
    
            pad_axis_0_labels = (int(pad_size[0] / 2), int(pad_size[0] / 2))
            if pad_size[0] % 2 == 1:
                pad_axis_0_labels = (pad_axis_0_labels[0] + 1, pad_axis_0_labels[1])
    
            pad_axis_1_labels = (int(pad_size[1] / 2), int(pad_size[1] / 2))
            if pad_size[1] % 2 == 1:
                pad_axis_1_labels = (pad_axis_1_labels[0] + 1, pad_axis_1_labels[1])
    
            labels = np.pad(labels, pad_width=(pad_axis_0_labels, pad_axis_1_labels))
            # data = np.pad(data, pad_width=((0, 0), pad_axis_0_labels, pad_axis_1_labels))
    
            if file not in invalid_scenes and file not in val_scenes:
                train_labels.append(labels)
                # train_data.append(data)
                count += 1
                print(count)
    
    uni_label = np.unique(train_labels)
    uni_label = uni_label[uni_label > 0]
    train_flat = np.array(train_labels).flatten()
    train_flat = train_flat[train_flat > 0]

    loss_weights = class_weight.compute_class_weight('balanced', uni_label,
                                                     train_flat)
    return loss_weights