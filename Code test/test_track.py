#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:53:41 2022

@author: alexis
"""


#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

import parallel_functions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="test3DdumpRaw"
o = osiris.Osiris(run,spNorm=None)

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (sx,sy,sz)

x     = o.getAxis("x")[sx]
y     = o.getAxis("y")[sy]
z     = o.getAxis("z")[sz]
extent=(min(x),max(x),min(y),max(y))

st = slice(None)
time = o.getTimeAxis(raw=True)[st]

#----------------------------------------------
data = o.getRaw(time, "eL", "tag", parallel=False)



