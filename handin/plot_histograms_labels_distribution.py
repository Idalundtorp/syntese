#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:53:05 2021

@author: s174020
"""

from sklearn.utils import class_weight
import numpy as np
import os
import xarray as xr
import torch

directory = '/work3/s174020/data/datasplit/CNN_data/augmented_weighted/'

train_labels_aug = []
test_labels = []
train_data = []
count = 0
# directory = '/work3/s174020/data/datasplit/CNN_data/'
val_scenes = np.genfromtxt('/work3/s174020/data/datasplit/CNN_data/' + 'validation_scenes.txt', dtype=str,
                           skip_header=2)
invalid_scenes = np.genfromtxt(directory + 'invalid_scenes.txt', dtype=str,
                               skip_header=2)
for file in os.listdir(directory):
    if 'ncdone' in file:
        fn = os.path.join(directory, file)
        ds = xr.open_dataset(fn)
        ndarr = ds.to_array()
        labels = ndarr[17, :, :].values  # SIC modes
        data = ndarr[:14, :, :].values   # all other channels expect SIC means
        size = (256, 256)
        shape = np.shape(labels)

        pad_size = (size[0]-shape[0], size[1]-shape[1])
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


        if file not in invalid_scenes and file not in val_scenes:
            train_labels_aug = np.append(train_labels_aug, labels)
            count += 1
            print(count)
        elif file not in invalid_scenes and file in val_scenes:
            test_labels = np.append(test_labels, labels)

directory = '/work3/s174020/data/datasplit/CNN_data/'

train_labels_all = []
train_data = []
count = 0
# directory = '/work3/s174020/data/datasplit/CNN_data/'
val_scenes = np.genfromtxt('/work3/s174020/data/datasplit/CNN_data/' + 'validation_scenes.txt', dtype=str,
                           skip_header=2)
invalid_scenes = np.genfromtxt(directory + 'invalid_scenes.txt', dtype=str,
                               skip_header=2)
for file in os.listdir(directory):
    if '.ncdone' in file:
        fn = os.path.join(directory, file)
        ds = xr.open_dataset(fn)
        ndarr = ds.to_array()
        labels = ndarr[17, :, :].values  # SIC modes
        data = ndarr[:14, :, :].values   # all other channels expect SIC means
        size = (256, 256)
        shape = np.shape(labels)

        pad_size = (size[0]-shape[0], size[1]-shape[1])
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


        if file not in invalid_scenes and file not in val_scenes:
            train_labels_all = np.append(train_labels_all, labels)
            count += 1
            print(count)
        # elif file not in invalid_scenes and file in val_scenes:
        #     test_labels = np.append(test_labels, labels)

#%%
import matplotlib.pyplot as plt
import numpy as np
uni = np.unique(train_labels_aug, return_counts=True)
uni_label = uni[0]
counts_aug = uni[1]
uni_label = uni_label[uni_label > 0]
train_flat = np.array(train_labels_aug).flatten()
train_flat_aug = train_flat[train_flat > 0]

uni = np.unique(train_labels_all, return_counts=True)
uni_label = uni[0]
counts_all = uni[1]
uni_label = uni_label[uni_label > 0]
train_flat = np.array(train_labels_all).flatten()
train_flat_all = train_flat[train_flat > 0]

conc = ['<10%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '95%<']

fig = plt.figure(figsize=(9, 4)) # Make it 14x7 inch
ax = fig.add_subplot(111)

plt.style.use('seaborn-whitegrid') # nice and clean grid
ax.bar(uni_label-0.15, counts_all[1:], color='#0504aa', width=0.2, align='center', alpha=0.6, linewidth=0.5)  # density=False would make counts
ax.bar(uni_label+0.15, counts_aug[1:], color='red', width=0.2, alpha=0.6, linewidth=0.5)  # density=False would make counts
ax.set_xticks(uni_label)
ax.set_xticklabels(conc)
plt.ylabel('Number of samples', fontsize=13)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('SIC categories (labels)', fontsize=13)
plt.title('Distribution of training data', fontsize=16)

uni_test = np.unique(test_labels, return_counts=True)
uni_label_test = uni_test[0]
counts_test = uni_test[1]
uni_label_test = uni_label_test[uni_label_test > 0]
test_flat = np.array(test_labels).flatten()
test_flat = test_flat[test_flat > 0]

# plt.figure(figsize=(9, 4)) # Make it 14x7 inch
# plt.style.use('seaborn-whitegrid') # nice and clean grid
# plt.bar(conc, counts_test[1:], color='#0504aa', alpha=0.8, linewidth=0.5)  # density=False would make counts
# plt.xticks(rotation=45, fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel('Number of samples', fontsize=13)
# plt.xlabel('SIC categories (labels)', fontsize=13)
# plt.title('Distribution of testing data', fontsize=16)


# plt.style.use('seaborn-whitegrid') # nice and clean grid
# ax.bar(uni_label-0.25, counts_all[1:], color='#0504aa', width=0.2, align='center', alpha=0.4, linewidth=0.5)  # density=False would make counts
# ax.bar(uni_label, counts_aug[1:], color='#0504aa', width=0.2, alpha=0.8, linewidth=0.5)  # density=False would make counts
# ax.bar(uni_label+0.25, counts_test[1:], color='#0504aa', width=0.2, alpha=0.8, linewidth=0.5)  # density=False would make counts
# plt.ylabel('Number of samples', fontsize=13)
# plt.xticks(rotation=45, fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('SIC categories (labels)', fontsize=13)
# plt.title('Distribution of training data', fontsize=16)