#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:36:37 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import parallelFunctions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

sx=slice(None)
sy = 0
sl = (sx,sy)

st = 52
x    = o.getAxis("x")
y    = o.getAxis("y")
time = o.getTimeAxis()

species='iL'
smooth = True

#----------------------------------------------
mu = np.abs(o.rqm[o.sIndex(species)])
n0 = o.n0[o.sIndex(species)]

Ex = o.getE(time[st], "x", sl=sl)
niL = np.abs(o.getCharge(time[st], species, sl=sl))
TiLx = o.getUfluid(time[st], species, "x", sl=sl)**2 * mu

if smooth:
    from scipy import signal

    std= 2

    win=signal.gaussian(len(x),std)   #window
#        gradPxy[j] = signal.convolve(gradPxy[j], win, mode='same') / sum(win)
    Ex = signal.convolve(Ex, win, mode='same') / sum(win)
    niL = signal.convolve(niL, win, mode='same') / sum(win)
    TiLx = signal.convolve(TiLx, win, mode='same') / sum(win)



#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(x,Ex-np.mean(Ex),color="b")
sub1.plot(x,niL-np.mean(niL),  color="k")
sub1.plot(x,(TiLx-np.mean(TiLx))/10, color="g")



#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)


sub1.set_ylim(-1,1)



dx = x[1]-x[0]

sub1.plot(x,np.gradient(Ex,dx)/5,color="b")
sub1.plot(x,np.gradient(niL,dx)/10,  color="k")
sub1.plot(x,np.gradient(TiLx,dx)/100, color="g")

sub1.fill_between(x,np.gradient(niL,dx)/10,np.gradient(TiLx,dx)/100,color="silver")
