#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:08:31 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS3Drmhr"
o = osiris.Osiris(run,spNorm="iL")
species ="iL"

mu = o.rqm[o.sIndex(species)]

st = slice(None)
time = o.getTimeAxis()[st]
N = len(time)
dt = time[1]-time[0]

#----------------------------------------------
#window is fixed along x axis, on depth fixed along z axis, moves along y axis
dx = 20
dy = 20
dz = 20

start_x = 5
end_x   = start_x+dx
sx = slice(start_x,end_x,1)

start_z = 0
end_z   = start_z+dz
sz = slice(start_z,end_z,1)

start_y = 0
end_y   = start_y+dy

v_window = 1   #number of c/wpi  per wpi^{-1}
incr = int(v_window*dt)

#----------------------------------------------
data = np.zeros(N)
for i in range(N):

    sy = slice(start_y,end_y,1)

    sl = (sx,sy,sz)
    data[i] = o.getCharge(time[i], species, sl=(sx,sz,sz),av=(1,2,3))

    start_y += incr
    end_y   += incr



#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,data)
sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
