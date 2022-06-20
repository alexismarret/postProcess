#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:19:18 2022

@author: alexis
"""
#----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import osiris
import h5py

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
# run ="CS3Dtrack"
run="testTrackSingle"

# spNorm = "eL"
spNorm="eL"
o = osiris.Osiris(run,spNorm=spNorm)

species ="eL"

st = slice(None,None,10)
sx = slice(None,None,1)
sp = slice(None,None,1)
sl = (sx,sp)

time = o.getTimeAxis(pha=True)[st]

pha = o.getPhaseSpace(time, species, "x", "x", sl=sl)
boundX, boundY = o.getBoundPhaseSpace(species)

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
# fig.subplots_adjust(bottom=0.28)

extent=(boundX[0],boundX[1],boundY[0],boundY[1])

im=sub1.imshow(pha[-1,...].T,
                extent=extent,origin="lower",
               aspect=1,
               cmap="bwr",
               vmin = -0.05, vmax = 0.05,
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)




