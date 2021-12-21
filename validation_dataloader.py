#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Dataloader to get validation/testing data"""

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

# -- Proprietary modules -- #



class ASIPDatasetVal(Dataset):
    """Class to create ASIP dataset."""

    def __init__(self, directory, transform=None, target_transform=None):
        # directory='/work3/s174020/data/datasplit/CNN_data'
        self.val_labels = []
        self.val_data = []
        self.ID = []
        self.index_nan_labels = []
        self.index_nan_data = []
        self.lat = []
        self.lon = []
        self.dir = directory
        # always validating on the same scenes independent on data augmentation
        val_scenes = np.genfromtxt('/work3/s174020/data/datasplit/CNN_data/validation_scenes.txt', dtype=str
                                   , skip_header=2)
        count = 0
        for file in os.listdir(directory):
            fn = os.path.join(directory, file)
            if '.ncdone' in file:
                if file in val_scenes:
                    # print(file)
                    ID = file.replace('_AMSR2_Icechart', '')
                    ID = ID.replace('_sub.ncdone', '')
                    ds = xr.open_dataset(fn)
                    ndarr = ds.to_array()
                    ind = np.isnan(ndarr[14, :, :].values)
                    labels = ndarr[17, :, :].values  # SIC modes
                    data = ndarr[:16, :, :].values   # all other channels expect SIC means
                    lat = ndarr[18, :, :].values
                    lon = ndarr[19, :, :].values
                    #data = np.delete(data, [2, 3], axis=0)  # remove 23.8 GHz
                    size = (256, 256)
                    # size = (128, 128)
                    # labels = np.resize(labels, size)
                    # data = np.resize(data, (16, 128, 128))
                    shape = np.shape(labels)
                    # print(shape)
                    pad_size = (size[0]-shape[0], size[1]-shape[1])
                    # print(pad_size)

                    for i in range(len(data[:, 0, 0])):
                        data[i][ind] = 0
                    
                    data[np.isnan(data)] = 0

                    
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
                    # labels = np.pad(labels, pad_width=((0, pad_size[0]), (0, pad_size[1])))
                    # data = np.pad(data, pad_width=((0, 0), (0, pad_size[0]), (0, pad_size[1])))
                    # mask = np.ones((shape))
    
                    pad_axis_0_labels = (int(pad_size[0] / 2), int(pad_size[0] / 2))
                    if pad_size[0] % 2 == 1:
                        pad_axis_0_labels = (pad_axis_0_labels[0] + 1, pad_axis_0_labels[1])
    
                    pad_axis_1_labels = (int(pad_size[1] / 2), int(pad_size[1] / 2))
                    if pad_size[1] % 2 == 1:
                        pad_axis_1_labels = (pad_axis_1_labels[0] + 1, pad_axis_1_labels[1])
    
                    labels = np.pad(labels, pad_width=(pad_axis_0_labels, pad_axis_1_labels))
                    lat = np.pad(lat, pad_width=(pad_axis_0_labels, pad_axis_1_labels))
                    lon = np.pad(lon, pad_width=(pad_axis_0_labels, pad_axis_1_labels))
                    # mask = np.array(np.pad(mask, pad_width=(pad_axis_0_labels, pad_axis_1_labels)), dtype=bool)
                    # labels = np.pad(labels, path_with)
                    data = np.pad(data, pad_width=((0, 0), pad_axis_0_labels, pad_axis_1_labels))
    
                    index_nan_labels = labels == 0  # gives both padding and land
                    index_nan_data = data[0, :, :] == 0

                    count += 1
                    print(count)
                    # print(file)
                    self.ID.append(ID)
                    self.index_nan_labels.append(index_nan_labels)
                    self.index_nan_data.append(index_nan_data)
                    self.lat.append(lat)
                    self.lon.append(lon)
                    self.val_labels.append(labels)
                    self.val_data.append(data)

    def __len__(self):
        """Define length of labels tensor."""
        return len(self.val_labels)

    def __getitem__(self, idx):
        """Get data based on batchsize defiend by idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # labels=torch.tensor(self.labels) #convert to labels tensor
        # data=torch.tensor(self.data) #convert to tensor
        ID = self.ID[idx]
        data = self.val_data[idx]
        labels = self.val_labels[idx]
        index_nan_labels = self.index_nan_labels[idx]
        index_nan_data = self.index_nan_data[idx]
        lat = self.lat[idx]
        lon = self.lon[idx]
        return data, labels, ID, index_nan_labels, index_nan_data, lat, lon
