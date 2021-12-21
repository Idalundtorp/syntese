#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:18:47 2021

@author: s174020

Distance calculation


"""
def convert_dist(dist):
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    import os,os.path
    
    
    #dist=distance_map
    dist_out=np.zeros(np.shape(dist))
    dist_out[np.round(dist)==0]=0
    dist_out[np.round(dist)==1]=0.25 #less that 10%
    dist_out[np.round(dist)==2]=0.75 #ocean water
    dist_out[np.round(dist)==3]=1.5
    dist_out[np.round(dist)==4]=2.5
    dist_out[np.round(dist)==5]=3.5
    dist_out[np.round(dist)==6]=4.5
    dist_out[np.round(dist)==7]=5.5
    dist_out[np.round(dist)==8]=6.5
    dist_out[np.round(dist)==9]=7.5
    dist_out[np.round(dist)==10]=8.5
    dist_out[np.round(dist)==11]=9.5
    dist_out[np.round(dist)==12]=10.5
    dist_out[np.round(dist)==13]=11.5
    dist_out[np.round(dist)==14]=12.5
    dist_out[np.round(dist)==15]=13.5
    dist_out[np.round(dist)==16]=14.5
    dist_out[np.round(dist)==17]=15.5
    dist_out[np.round(dist)==18]=16.5
    dist_out[np.round(dist)==19]=17.5
    dist_out[np.round(dist)==20]=18.5
    dist_out[np.round(dist)==21]=19.5
    
    dist_out[np.round(dist)==22]=22.5
    dist_out[np.round(dist)==23]=27.5
    dist_out[np.round(dist)==24]=32.5
    dist_out[np.round(dist)==25]=37.5
    dist_out[np.round(dist)==26]=42.5
    dist_out[np.round(dist)==27]=47.5
    
    # from polar_plots import plot
    # extent=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    # fig, ax = plt.subplots()
    # plot(lat,lon,dist,extent,[0,80])
    # plt.title('SIC-AMSR2-grid', loc='left')
    # #plt.savefig('dist-AMSR2-grid'+area+'.jpg')
    
    dist_out[np.round(dist)==28]=55
    dist_out[np.round(dist)==29]=65
    dist_out[np.round(dist)==30]=75
    
    # from polar_plots import plot
    # extent=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    # fig, ax = plt.subplots()
    # plot(lat,lon,dist,extent,[0,80])
    # plt.title('SIC-AMSR2-grid', loc='left')
    
    dist_out[np.round(dist)==31]=85
    dist_out[np.round(dist)==32]=95
    
    dist_out[np.round(dist)==33]=112.5
    dist_out[np.round(dist)==34]=137.5
    dist_out[np.round(dist)==35]=162.5
    dist_out[np.round(dist)==36]=187.5
    dist_out[np.round(dist)==37]=212.5
    
    dist_out[np.round(dist)==38]=237.5
    dist_out[np.round(dist)==39]=262.5
    dist_out[np.round(dist)==40]=287.5
    dist_out[np.round(dist)==41]=np.nan
    
    dist_out[np.isnan(dist)]=np.nan
        
    #0; land
    #1; 0 -> 0.5 km
    #2; 0.5 -> 1 km
    #3; 1 -> 2 km
    #4; 2 -> 3 km
    #5; 3 -> 4 km
    # 6; 4 -> 5 km
    # 7; 5 -> 6 km
    # 8; 6 -> 7 km
    # 9; 7 -> 8 km
    # 10; 8 -> 9 km
    # 11; 9 -> 10 km
    # 12; 10 -> 11 km
    # 13; 11 -> 12 km
    # 14; 12 -> 13 km
    # 15; 13 -> 14 km
    # 16; 14 -> 15 km
    # 17; 15 -> 16 km
    # 18; 16 -> 17 km
    # 19; 17 -> 18 km
    # 20; 18 -> 19 km
    # 21; 19 -> 20 km
    
    # 22; 20 -> 25 km
    # 23; 25 -> 30 km
    # 24; 30 -> 35 km
    # 25; 35 -> 40 km
    # 26; 40 -> 45 km
    # 27; 45 -> 50 km
    
    # 28; 50 -> 60 km
    # 29; 60 -> 70 km
    # 30; 70 -> 80 km
    # 31; 80 -> 90 km
    # 32; 90 -> 100 km
    
    # 33; 100 -> 125 km
    # 34; 125 -> 150 km
    # 35; 150 -> 175 km
    # 36; 175 -> 200 km
    # 37; 200 -> 225 km
    # 38; 225 -> 250 km
    # 39; 250 -> 275 km
    # 40; 275 -> 300 km
    
    # 41; 300 -> inf km
    return dist_out
