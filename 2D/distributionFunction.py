#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:14:50 2022

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
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrRaw"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
st = slice(None,None,10)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sy]
time = o.getTimeAxis(species="iL", raw=True)[st]

#['SIMULATION', 'ene', 'p1', 'p2', 'p3', 'q', 'tag', 'x1', 'x2', 'x3']

#----------------------------------------------

cells = np.array([[25,50],[25,50]])

for i in range(len(time)):

    x1 = o.getRaw(time[i], "iL", "x1")
    x2 = o.getRaw(time[i], "iL", "x2")

    pos = (x1,x2)
    gi, gj = o.findCell(pos)

    for k in range(len(gi)):
        for l in range(len(gj)):

            cond = ((gi >= cells[0,0]) & (gi <= cells[0,1]) &
                    (gj >= cells[1,0]) & (gj <= cells[1,1]))

            gic = np.nozero(gi[cond])
            gjc = np.nozero(gj[cond])




# vPart_iLx = o.getRaw(time, "iL", "p1")

# for i in range(len(vPart_iLx)):print(vPart_iLx[i].shape)
