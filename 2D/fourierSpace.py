#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:33:01 2022

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
        'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS2Drmhr"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,1)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis()[st]

#----------------------------------------------
# Ex = o.getE(time,"x")
# Bz = o.getB(time,"z")

axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0])[1:] *2*np.pi
axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0])[1:] *2*np.pi

# fb = np.abs(np.fft.rfft(Bz,axis=2)[:,:,1:])
# fe = np.abs(np.fft.rfft(Ex,axis=2)[:,:,1:])

ftB = np.mean(
              np.abs(
                  np.fft.rfft(o.getB(time,"z"),axis=2)[:,:,1:]
                  ),
              axis=1)
# ftE = np.mean(fe,axis=1)
# std_ftB = np.std(fb,axis=1)
# std_ftE = np.std(fe,axis=1)

extent=(min(axis_kY),max(axis_kY),min(time),max(time))
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.set_xscale('log')
# sub1.set_yscale('log')

eps = 1e-7
im=sub1.imshow(ftB + eps,   #no .T to keep first axis (time) along y
               extent=extent,origin="lower",
                aspect="auto",
               cmap="jet",
                norm=LogNorm(vmin=1e-2,vmax=1e2),
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

# sub1.locator_params(nbins=5,axis='y')
# sub1.locator_params(nbins=5,axis='x')

sub1.set_xlabel(r'$k\ [\omega_{pi}/c]$')
sub1.set_ylabel(r'$t\ [\omega_{pi}^{-1}]$')

sub1.set_xlim(min(axis_kY),max(axis_kY))


"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

for t in range(len(time)):
    sub1.loglog(axis_kY, ftB[t])
    # sub1.fill_between(axis_kY, ftB[t]-std_ftB[t],
    #                            ftB[t]+std_ftB[t],alpha=0.3)

sub1.set_xlabel(r"$k_y\ [\omega_{pi}/c]$")
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

for t in range(len(time)):
    sub1.loglog(axis_kY, ftE[t])
    # sub1.fill_between(axis_kY, ftE[t]-std_ftE[t],
    #                            ftE[t]+std_ftE[t],alpha=0.3)
"""
