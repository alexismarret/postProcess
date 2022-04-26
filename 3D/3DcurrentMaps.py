#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:10:53 2022

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
# plt.close("all")

#----------------------------------------------
run  ="CS3D"
o = osiris.Osiris(run)

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (sx,sy,sz)

x     = o.getAxis("x")[sx]
y     = o.getAxis("y")[sy]
z     = o.getAxis("z")[sz]

st = slice(None)
st=3
time = o.getTimeAxis()[st]

qiL = o.getCharge(time, "iL", sl=sl, av=0,parallel=False)


print(qiL.shape)
