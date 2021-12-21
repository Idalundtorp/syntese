#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:03:11 2021

@author: s174020

"""
def plot(lat,lon,z,extent,clim):
    import cartopy.crs as ccrs
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import cartopy.feature as cfeature

    #ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-50))
    ax = plt.axes(projection=ccrs.Mercator(central_longitude=-50))
    # ax.stock_img()
    ax.coastlines()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 14, 'color': 'black'}
    gl.ylabel_style = {'size': 14, 'color': 'black'}
    # gl.ylabels_left = True
    # gl.xlines = False
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style = {'size': 15, 'color': 'gray'}
    # gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    # extent=[-80,0,60,90]
    ax.set_extent(extent, ccrs.PlateCarree())
    # ax.gridlines(draw_labels=True)

    plot = plt.scatter(lon, lat,
             s=10, c=z, alpha=1, cmap=plt.get_cmap('Blues_r'),
             transform=ccrs.Geodetic(),)

    # sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
    plt.clim(clim[0], clim[1])
    # plt.clim(min(z),max(z))
    cb = plt.colorbar(plot)
    cb.ax.tick_params(labelsize=14)
    ax.add_feature(cfeature.LAND, zorder=100, facecolor='black')
    ax.gridlines()
    ax.set_facecolor('lightgrey')
    # ax.add_feature(cfeature.OCEAN)        

    #theta = np.linspace(0, 2*np.pi, 100)
    #center, radius = [0.5, 0.5], 0.5
    #verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    #circle = mpath.Path(verts * radius + center)
    #ax.set_boundary(circle, transform=ax.transAxes)
    