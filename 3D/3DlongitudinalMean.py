#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:02:52 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt
import fit

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="CS3D_noKink"
run = "CS3Dtrack"

o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
# sz = slice(None,None,1)
sz = 0

sl = (sx,sy,sz)
av = 1

st = slice(0,None,1)
time = o.getTimeAxis()[st]

y = o.getAxis("x")[sy]
Ex = o.getE(time, "x", sl=sl, av=av, parallel=False)
Uix = o.getUfluid(time, "iL", "x", sl=sl, av=av, parallel=False)

#%%

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

extent=(min(y),max(y),min(time),max(time))
im=sub1.imshow(Ex,
               extent=extent,origin="lower",
                aspect=0.05,
               cmap="bwr",
                vmin = -0.01, vmax = 0.01,
               interpolation="None")

sub1.set_xlabel(r"$y\ [c/\omega_{pi}]$")
sub1.set_ylabel(r"$t\ [\omega_{pi}^{-1}]$")


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

extent=(min(y),max(y),min(time),max(time))
im=sub1.imshow(Uix,
               extent=extent,origin="lower",
                aspect=0.05,
               cmap="bwr",
                vmin = 0.1, vmax = 0.7,
               interpolation="None")

sub1.set_xlabel(r"$y\ [c/\omega_{pi}]$")
sub1.set_ylabel(r"$t\ [\omega_{pi}^{-1}]$")


