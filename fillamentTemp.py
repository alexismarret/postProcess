#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:34:26 2022

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
        'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="counterStream"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,1)
x     = o.getAxis("x")[sx]
y     = o.getAxis("y")[sx]
time = o.getTimeAxis("eL")[st]

#----------------------------------------------
mask = o.locFilament(time, fac=2)

TiLx = o.getUth(time, "iL", "x")
TiLy = o.getUth(time, "iL", "y")

#f: value in filament
#nf: value out of filament
TiLx_f  = np.ma.mean(np.ma.masked_array(TiLx,mask= mask,copy=False),
                  axis=(1,2))
TiLx_nf = np.ma.mean(np.ma.masked_array(TiLx,mask=~mask,copy=False),
                  axis=(1,2))

TiLy_f  = np.ma.mean(np.ma.masked_array(TiLy,mask= mask,copy=False),
                  axis=(1,2))
TiLy_nf = np.ma.mean(np.ma.masked_array(TiLy,mask=~mask,copy=False),
                  axis=(1,2))

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(time,TiLx_f/TiLx_nf,color="r"     ,label=r"$T_{ix}^*/T_{ix}$")
sub1.plot(time,TiLy_f/TiLy_nf,color="orange",label=r"$T_{iy}^*/T_{iy}$")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_xlim(min(time),max(time))

sub1.legend(frameon=False)



#----------------------------------------------
eps=1e-5
r = o.getUth(time, "iL", "x") / (o.getUth(time, "iR", "x")+eps)

mask = np.ma.getmask(np.ma.masked_where(r<1,r,copy=False))

r_g1 = np.ma.mean(np.ma.masked_array(r,mask= mask,copy=False),axis=(1,2))
r_l1 = np.ma.mean(np.ma.masked_array(r,mask=~mask,copy=False),axis=(1,2))

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(time,r_g1,color="r",label=r"$\langle T_{iL}/T_{iR}>1\rangle$")
sub1.plot(time,r_l1,color="b",label=r"$\langle T_{iL}/T_{iR}<1\rangle$")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_xlim(min(time),max(time))

sub1.legend(frameon=False)
