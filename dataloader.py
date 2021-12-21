#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Dataloader to get training data"""

# -- File info --#
__author__ = 'Ida L. Olsen'
__contributors__ = ''
__contact__ = ['s174020@student.dtu.dk']
__version__ = '0'
__date__ = '2021-10-08'

# -- Built-in modules -- #
import os

# -- Third-part modules -- #
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
from sklearn.utils import class_weight

# -- Proprietary modules -- #


class ASIPDataset(Dataset):
    """Class to create ASIP dataset."""

    def __init__(self, directory, transform=None, target_transform=None):
        self.train_labels = []
        self.train_data = []
        self.ID = []
        val_scenes = np.genfromtxt(directory + 'validation_scenes.txt', dtype=str,
                                   skip_header=2)
        invalid_scenes = np.genfromtxt(directory + 'invalid_scenes.txt', dtype=str,
                                       skip_header=2)
        self.dir = directory
        for file in os.listdir(directory):
            if '.ncdone' in file:
                fn = os.path.join(directory, file)
    
                ID = file.replace('_AMSR2_Icechart', '')
                ID = ID.replace('_sub.ncdone', '')
    
                ds = xr.open_dataset(fn)
                ndarr = ds.to_array()
                ind = np.isnan(ndarr[14, :, :].values)
                labels = ndarr[17, :, :].values  # SIC modes
                data = ndarr[:16, :, :].values   # all other channels expect SIC means
                #data = np.delete(data, [2, 3], axis=0)  # remove 23.8 GHz
                size = (256, 256)
                shape = np.shape(labels)
                # print(shape)
                pad_size = (size[0]-shape[0], size[1]-shape[1])
                # print(pad_size)
                # labels = np.resize(labels, size)
                # data = np.resize(data, (16, 128, 128))
                #ind = np.isnan(data)[14]
                for i in range(len(data[:, 0, 0])):
                    #print(i)
                    data[i][ind] = 0
                    # print(data.shape)
                    #print(data[i][ind])
                
                data[np.isnan(data)] = 0
                #print(np.where(np.isnan(data)))

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

                labels[ind] = 0

                pad_axis_0_labels = (int(pad_size[0] / 2), int(pad_size[0] / 2))
                if pad_size[0] % 2 == 1:
                    pad_axis_0_labels = (pad_axis_0_labels[0] + 1, pad_axis_0_labels[1])
    
                pad_axis_1_labels = (int(pad_size[1] / 2), int(pad_size[1] / 2))
                if pad_size[1] % 2 == 1:
                    pad_axis_1_labels = (pad_axis_1_labels[0] + 1, pad_axis_1_labels[1])
    
                labels = np.pad(labels, pad_width=(pad_axis_0_labels, pad_axis_1_labels))
                # labels = np.pad(labels, path_with)
                data = np.pad(data, pad_width=((0, 0), pad_axis_0_labels, pad_axis_1_labels))
    
                if file not in invalid_scenes and file not in val_scenes:
    
                    self.ID.append(ID)
                    self.train_labels.append(labels)
                    self.train_data.append(data)
    
    def __len__(self):
        """Define length of labels tensor."""
        return len(self.train_labels)

    def __getitem__(self, idx):
        """Get data based on batchsize defiend by idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ID = self.ID[idx]
        data = self.train_data[idx]
        labels = self.train_labels[idx]

        return data, labels, ID
