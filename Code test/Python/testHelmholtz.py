#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:37:03 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
# run  = ["CS3D_noKink","CS3Dtrack"]
# run = ["CS3Drmhr","CS3Drmhr"]
run = ["CS2DrmhrTrack"]
o = osiris.Osiris(run[0],spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)

y = o.getAxis("y")[sx]

# sl = (sx,sy,sz)
sl = (0,sy)

# ax = (0,1,2)
ax = (0,1)

st = slice(None)
time = o.getTimeAxis()[st]

t = 15

#----------------------------------------------
Ex  = o.getE(time[t], "x", sl=sl)
Ey  = o.getE(time[t], "y", sl=sl)
Ez  = o.getE(time[t], "z", sl=sl)

Ecx = o.getNewData(time[t], "Ecx", sl=sl)
Ecy = o.getNewData(time[t], "Ecy", sl=sl)
Ecz = o.getNewData(time[t], "Ecz", sl=sl)

Erx = Ex-Ecx
Ery = Ey-Ecy
Erz = Ez-Ecz


#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(y,Ex,color="k")
sub1.plot(y,Ecx,color="g",linestyle="--")
sub1.plot(y,Erx,color="r",linestyle="--")

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(y,Ey,color="k")
sub1.plot(y,Ecy,color="g",linestyle="--")
sub1.plot(y,Ery,color="r",linestyle="--")

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(y,Ez,color="k")
sub1.plot(y,Ecz,color="g",linestyle="--")
sub1.plot(y,Erz,color="r",linestyle="--")


