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

import parallel_functions as pf
from scipy import signal

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="counterStreamFast"
o = osiris.Osiris(run,spNorm="iL")

x     = o.getAxis("x")
y     = o.getAxis("y")
time = o.getTimeAxis("eL")

xpos = 0
t = 1
time = time[t]

#----------------------------------------------
B = o.getB(time,"z")[0,xpos]
E = o.getE(time,"x")[0,xpos]

j_iLx = (o.getCurrent(time, "eL", "x")[0,xpos]+
         o.getCurrent(time, "eR", "x")[0,xpos]+
         o.getCurrent(time, "iL", "x")[0,xpos]+
         o.getCurrent(time, "iR", "x")[0,xpos])
# j_iLx = o.getCurrent(time, "iL", "x")[0,xpos]

R=len(x)
st=2.     #standard deviation for gaussian filter
win=signal.gaussian(R,st)   #window

B      = signal.convolve(B, win, mode='same') / sum(win)
E     = signal.convolve(E, win, mode='same') / sum(win)
j_iLx  = signal.convolve(j_iLx, win, mode='same') / sum(win)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,1.8),dpi=300)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(y,E,color="r",label=r"$E_x$")
sub1.plot(y,B,color="g",label=r"$B_z$")
sub1.plot(y,j_iLx ,color="b",label=r"$J_{x}$")



sub1.legend(frameon=False)
# sub1.set_xlim(time[0],time[-1])
# sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
# sub1.set_ylabel(r"$(\mathcal{E}-\mathcal{E}_0)/\mathcal{E}_0$")
