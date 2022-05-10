#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:26:32 2022

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
from scipy import signal

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS2Drmhr"
o = osiris.Osiris(run,spNorm=None)

x     = o.getAxis("x")
y     = o.getAxis("y")          #7-12
time = o.getTimeAxis("eL")

xpos = 0
t = 1
time = time[t]
filtr = True
#----------------------------------------------
Bx = o.getB(time,"x")[xpos]
By = o.getB(time,"y")[xpos]
Bz = o.getB(time,"z")[xpos]

Ex = o.getE(time,"x")[xpos]
Ey = o.getE(time,"y")[xpos]
Ez = o.getE(time,"z")[xpos]

j_iLx = (o.getCurrent(time, "eL", "x")[xpos]+
         o.getCurrent(time, "eR", "x")[xpos]+
         o.getCurrent(time, "iL", "x")[xpos]+
         o.getCurrent(time, "iR", "x")[xpos])

if filtr:
    st=2.     #standard deviation for gaussian filter
    win=signal.gaussian(len(y),st)   #window

    Bx      = signal.convolve(Bx, win, mode='same') / np.sum(win)
    By      = signal.convolve(By, win, mode='same') / np.sum(win)
    Bz      = signal.convolve(Bz, win, mode='same') / np.sum(win)

    Ex      = signal.convolve(Ex, win, mode='same') / np.sum(win)
    Ey      = signal.convolve(Ey, win, mode='same') / np.sum(win)
    Ez      = signal.convolve(Ez, win, mode='same') / np.sum(win)

    j_iLx   = signal.convolve(j_iLx, win, mode='same') / np.sum(win)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,1.8),dpi=300)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(y,j_iLx ,color="b",label=r"$J_{x}$")
sub1.plot(y,Bz,color="g",label=r"$B_z$")

sub1.plot(y,Ex,color="r",label=r"$E_x$")
# sub1.plot(y,Ey,color="orange",label=r"$E_y$")


sub1.legend(frameon=False)

sub1.set_xlabel(r'$y\ [l_0]$')

# sub1.legend(frameon=False)
# sub1.set_xlim(time[0],time[-1])
# sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
# sub1.set_ylabel(r"$(\mathcal{E}-\mathcal{E}_0)/\mathcal{E}_0$")
