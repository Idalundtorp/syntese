#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:34:12 2021

@author: s174020
"""
import numpy as np
import matplotlib.pyplot as plt


with open('loss_weight.npy', 'rb') as f:
    loss_weights = np.load(f)

with open('loss_weight_org.npy', 'rb') as f:
    loss_weights_org = np.load(f)

weights_org = np.insert(loss_weights_org, 0, 0)
weights = np.insert(loss_weights, 0, 0)

labels = np.linspace(0, 11, len(weights_org))
plt.figure(figsize=(7, 5))
plt.plot(labels, weights_org, label='Weights using sklearn method')
plt.plot(labels, weights, label='Weights using median freq')
plt.xlim(0, 11)
plt.ylim(0, 11)
plt.xlabel('labels')
plt.ylabel('weights')
plt.legend(loc='upper left')
plt.suptitle('Weights for each SIC label', fontsize=14)
plt.title('Factor difference between weighting methods: '+str(np.round(np.nanmedian(weights_org/weights), 2)),
          fontsize=12)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(labels, weights_org/(np.nanmedian(weights_org/weights)), label='Weights using sklearn method')
plt.plot(labels, weights, label='Weights using median freq')
plt.xlim(0, 11)
plt.xlabel('labels')
plt.ylabel('weights')
plt.suptitle('Weights for each SIC label', fontsize=14)
plt.title('Factor difference between weighting methods: '+str(np.round(np.nanmedian(weights_org/weights), 2)),
          fontsize=12)
plt.show()