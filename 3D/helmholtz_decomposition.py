#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:41:20 2022

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
import gc

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")


#----------------------------------------------
# run  ="CS3D_noKink"
run = "CS3Dtrack"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (sx,sy,sz)

x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sy]
z    = o.getAxis("z")[sz]

st = slice(None,None,1)
time = o.getTimeAxis()[st]

#----------------------------------------------
for i in range(len(time)):

    #empty memory before next loop to avoid 2x same array size in memory
    print(i)

    compr = o.helmholtzDecompose(x, y, z, comp=0, timeVal=time[i], sl=sl)
    o.writeHDF5(compr, "Ecx", timeArray=False, index=i)
    del compr
    gc.collect()

    # compr = o.helmholtzDecompose(x, y, z, comp=1)
    # o.writeHDF5(compr, "Ecx", timeArray=False, index=i)
    # del compr
    # gc.collect()

    # compr = o.helmholtzDecompose(x, y, z, comp=2)
    # o.writeHDF5(compr, "Ecx", timeArray=False, index=i)
    # del compr
    # gc.collect()


