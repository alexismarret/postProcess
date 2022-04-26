#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:33:01 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import parallel_functions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="CS2Drm"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,30)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis()[st]

#----------------------------------------------
Bz = o.getB(time,"z")
Ex = o.getE(time,"x")

axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0])[1:] *2*np.pi
axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0])[1:] *2*np.pi



ftB = np.zeros((len(time),len(axis_kY)))
ftE = np.zeros((len(time),len(axis_kY)))
std_ftB = np.zeros((len(time),len(axis_kY)))
std_ftE = np.zeros((len(time),len(axis_kY)))

for i in range(len(time)):
    fb = np.abs(np.fft.rfft(Bz[i],axis=1)[:,1:])
    fe = np.abs(np.fft.rfft(Ex[i],axis=1)[:,1:])

    ftB[i] = np.mean(fb,axis=0)
    ftE[i] = np.mean(fe,axis=0)
    std_ftB[i] = np.std(fb,axis=0)
    std_ftE[i] = np.std(fe,axis=0)



#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)


for t in range(len(time)):
    sub1.loglog(axis_kY, ftB[t])
    sub1.fill_between(axis_kY, ftB[t]-std_ftB[t],
                               ftB[t]+std_ftB[t],alpha=0.3)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)


for t in range(len(time)):
    sub1.loglog(axis_kY, ftE[t])
    sub1.fill_between(axis_kY, ftE[t]-std_ftE[t],
                               ftE[t]+std_ftE[t],alpha=0.3)
