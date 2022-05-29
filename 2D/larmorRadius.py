#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:01:29 2022

@author: alexis
"""


#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from scipy import signal
from scipy.stats import skew

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import parallelFunctions as pf
import time as ti

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run = "CS2DrmhrRawLall"
o = osiris.Osiris(run,spNorm="iL")

species = "iL"

st = slice(None,None,1)
x    = o.getAxis("x")
y    = o.getAxis("y")
time = o.getTimeAxis(species=species, raw=True)[st]

for i in range(len(time)):

    p1 = o.getRaw(time[i], species, "p1")
    p2 = o.getRaw(time[i], species, "p2")
    p3 = o.getRaw(time[i], species, "p3")
    lorentz = np.sqrt(1+p1**2+p2**2+p3**2)

    B1  = o.getB(time[i],"x")
    B2  = o.getB(time[i],"y")
    B3  = o.getB(time[i],"z")

    vperp1 = o.projectVec(p1, p2, p3, B1, B2, B3, 1)
    vperp2 = o.projectVec(p1, p2, p3, B1, B2, B3, 2)



