# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Script to run the CNN netowrk."""

# -- File info --#
__author__ = 'Ida L. Olsen'
__contributors__ = ''
__contact__ = ['s174020@student.dtu.dk']
__version__ = '0'
__date__ = '2021-10-08'

# -- Built-in modules -- #
# import sys
# sys.path.append('/work3/s174020/')

# -- Third-part modules -- #
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
from UNET_corr import UNet
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score as r2_loss

# -- Proprietary modules -- #
from dataloader import ASIPDataset
from Get_weights import Get_weights
from Get_weights_org import Get_weights_org
from validation_dataloader import ASIPDatasetVal
# from main_train_loop import Trainer
from Main_train_loop_original import Train_Loop
#%%
# directory to read data from
directory = '/work3/s174020/data/datasplit/CNN_data/augmented_weighted/'
# directory for saving predictions
save_dir = '/work3/s174020/plots/for_show/Report_07_12_2021/'
# name to identify test
name = '18,36,89_best'
# choose weighting technique
ignore_index = False
weights_org = True
weights_new = False
# choose LR
LR = 10**(-2)
# Choose number of EPOCHS
EPOCHS = 100

# make training and validation datasets
train_set = ASIPDataset(directory=directory)
val_set = ASIPDatasetVal(directory=directory)

image_datasets = {
    'train': train_set, 'val': val_set
}

# batch_size to avoid loading all data at once
batch_size = 64

# num_workers: kommer an på forholdet mellem batchsize og træningstid
dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3),
               'val': DataLoader(val_set, batch_size=1, shuffle=True)}

# Get a batch of training data
inputs, labels, ID = next(iter(dataloaders['train']))

print(inputs.shape, labels.shape)

# check if GPU is avalible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = inputs.cuda()

#%% 
# load the UNET model
def weight_reset(module):
    """Reset weights in model."""
    if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
        module.reset_parameters()


model = UNet(in_channels=16,
             out_channels=12,
             n_blocks=4,
             start_filters=16,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)
# Make the model run on the GPU
model = model.to(device)

# If using weights in cross entropy to adress label imbalance
if weights_new:
    loss_weights = Get_weights(directory)
    with open('loss_weight.npy', 'wb') as f:
        np.save(f, loss_weights)
    with open('loss_weight.npy', 'rb') as f:
        loss_weights = np.load(f)
elif weights_org:
    loss_weights_org = Get_weights_org(directory)
    with open('loss_weight_org.npy', 'wb') as f:
        np.save(f, loss_weights_org) 
    with open('loss_weight_org.npy', 'rb') as f:
        loss_weights = np.load(f)
elif ignore_index:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
if weights_new or weights_org:
    weights = np.insert(loss_weights, 0, 0)  # equivalent to ignore index = 0
    weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

# optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.6)
# from torch.optim.lr_scheduler import ExponentialLR, StepLR
# scheduler = StepLR(optimizer, gamma=1)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # , weight_decay=0.1)

out = model(inputs)
print(f'Out: {out.shape}')

# check a model summary using torchsummary
summary(model, input_size=(16, 256, 256))


H, val_loss_min, accuracy_best, epoch_min = Train_Loop(model=model,
                                                       device=device,
                                                       criterion=criterion,
                                                       optimizer=optimizer,
                                                       dataloaders=dataloaders,
                                                       EPOCHS=EPOCHS,
                                                       save_dir=save_dir,
                                                       LR=LR,
                                                       name=name)

#%% make predictions

from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV

# def r2_loss(output, target):
#     target_mean = torch.mean(target)
#     ss_tot = torch.sum((target - target_mean) ** 2)
#     ss_res = torch.sum((target - output) ** 2)
#     r2 = 1 - ss_res / ss_tot
#     return r2

EPOCH = epoch_min
# loading a saved model
model_name = "ASIP_model"+str(EPOCH)
model.load_state_dict(torch.load(model_name))
model.eval()
val_correct = 0
val_correct_org = 0
r2_score = 0

lab = torch.tensor(())
predictions = torch.tensor(())
with torch.no_grad():
    count = 0
    # loop over validation data
    for (inputs, labels, ID, mask, index_nan_data, lat, lon) in dataloaders['val']:
        ID = ID[0]
        
        count += 1
        
        labels = labels.cuda()
        inputs = inputs.cuda()
        pred = model(inputs)

        # find class label index with the largest probability (argmax takes largest prob)
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        # pred_label = pred.cpu().numpy()[0][idx]

        # base score on non zero class
        output = pred.argmax(axis=1).cpu()[0]
        ll = labels.cpu().squeeze()

        r2_score += r2_loss(ll[ll != 0].type(torch.double), output[ll != 0].type(torch.double))

        preds_corr = (idx[ll != 0] == ll[ll != 0].numpy())

        preds_corr_org = (pred.argmax(1) == labels.type(torch.float))

        val_correct_org += preds_corr_org.sum().item()
        # convert to binary
        val_correct += preds_corr.sum().item()/len(ll[ll != 0])

        idx = idx.astype(float)
        idx[idx == 0] = np.nan

        ## saving predictions and labels
        lab = torch.cat((lab, ll[ll != 0].type(torch.double).flatten()), 0)
        predictions = torch.cat((predictions, output[ll != 0].type(torch.double).flatten()), 0)


        # fig, ax = plt.subplots()
        labels = labels.cpu().numpy()[0]
        
        
        from polar_plots_pred import plot as pp
        index1_d = np.where(index_nan_data)[1]
        index2_d = np.where(index_nan_data)[2]
        # idx[index1_d, index2_d] = np.nan

        lat2 = lat.squeeze()
        lon2 = lon.squeeze()

        lat2[index1_d, index2_d] = np.nan
        lon2[index1_d, index2_d] = np.nan

        # lat2 = lat2.flatten()
        # lon2 = lon2.flatten()
        # idx = idx.flatten()

        clim = [1, 11]
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        plt.figure(figsize=(9, 6))
        pp(lat, lon, idx, extent, clim)
        plt.suptitle(ID, fontsize=16)
        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.savefig(save_dir+name+str(EPOCH)+str(LR)+str(count)+'pred_whole.png', bbox_inches='tight')
        plt.show()

        from polar_plots_pred import plot as pp
        index1 = np.where(mask)[1]
        index2 = np.where(mask)[2]
        labels[index1, index2] = np.nan

        lat1 = lat
        lon1 = lon

        lat1.squeeze()[index1, index2] = np.nan
        lon1.squeeze()[index1, index2] = np.nan
        lat1 = lat1.flatten()
        lon1 = lon1.flatten()

        plt.figure(figsize=(9, 6))
        pp(lat, lon, labels, extent, clim)
        plt.suptitle(ID, fontsize=16)
        plt.xlabel('latitude')
        plt.ylabel('longitude')

        plt.savefig(save_dir+name+str(EPOCH)+str(LR)+str(count)+'label.png', bbox_inches='tight')
        plt.show()


        plt.figure(figsize=(9, 6))
        pp(lat, lon, idx, extent, clim)
        plt.suptitle(ID, fontsize=16)
        plt.xlabel('latitude')
        plt.ylabel('longitude')

        plt.savefig(save_dir+name+str(EPOCH)+str(LR)+str(count)+'prediction.png', bbox_inches='tight')
        plt.show()

    # param_grid = dict(lr=[10**(-2), 5*10**(-2), 10**(-3), 5*10**(-3), 10**(-4)])
    # grid = GridSearchCV(estimator=model.fit(), param_grid=param_grid, n_jobs=-1, cv=3)
    # grid_result = grid.fit(lab, predictions)

    R2_score = r2_loss(lab, predictions)
    class_accuracy = confusion_matrix(lab, predictions, labels=np.unique(lab), normalize="true").diagonal()
    accuracy2_test = confusion_matrix(lab, predictions, labels=np.unique(lab), normalize="true").diagonal().sum()/len(np.unique(lab))
    balanced_accuracy = balanced_accuracy_score(lab, predictions)
    accuracy = (lab == predictions).sum()/len(lab)
    cm = confusion_matrix(lab, predictions, labels=np.unique(lab), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(lab))
    disp.plot(include_values=False, display_labels=[0, 1])
    
    # val_correct_org = val_correct_org / (len(dataloaders['val'].dataset)*len(ll.flatten()))  # accuracy per pixel
    # val_correct = val_correct / (len(dataloaders['val'].dataset))  # accuracy per pixel
    # avg_R2 = r2_score / (len(dataloaders['val'].dataset))

    print(R2_score)
    print(accuracy)
    print(accuracy2_test)
    print(balanced_accuracy)
    print(np.round(class_accuracy, 2))