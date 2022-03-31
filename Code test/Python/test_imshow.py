#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:48:03 2022

@author: alexis
"""

#----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import osiris

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
L = 75
NP = 50
x = np.linspace(0,100,NP)
y = np.linspace(0,300,NP*2)

f = np.exp(-np.sqrt(x[:,None]**2+y[None,:]**2)/L)

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,3),dpi=300)
# fig.subplots_adjust(bottom=0.28)

#by default, first dim goes in y axis, second dim goes in x axis
#if .T, it's the opposite: first dim in x, second in y
#extend needs to be adjusted: first two values are for x axis, other two for y axis

# extent=(min(timeP),max(timeP),min(x),max(x))
extent=(min(x),max(x),min(y),max(y))
im=sub1.imshow(f.T,
                extent=extent,
                origin="lower",
                aspect=1,
               cmap="hot",
               # vmin = 0.05, vmax = 0.15,
               interpolation="None")

#norm=LogNorm(vmin = 0.1, vmax = 10),
#aspect=1,
#extent=extent

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

sub1.locator_params(nbins=5,axis='y')
sub1.locator_params(nbins=5,axis='x')




#----------------------------------------------
run = "counterStream2"
o = osiris.Osiris(run)

key="charge"
species="iL"
dataPath = o.path+"/MS/DENSITY/" +species+"/"+key

def readScalar(dataPath,key):

    with h5py.File(dataPath,"r") as f:

        return f[key][()]

#----------------------------------------------
n = readScalar(dataPath+"/"+"charge-iL-000057.h5", key)
x      = o.getAxis("x")
y      = o.getAxis("y")
timeP  = o.getTimeParticles(species)

extent=(min(x),max(x),min(y),max(y))

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
# fig.subplots_adjust(bottom=0.28)

#by default, first dim goes in y axis, second dim goes in x axis
#if .T, it's the opposite: first dim in x, second in y
#extend needs to be adjusted: first two values are for x axis, other two for y axis

im=sub1.imshow(n.T,
               extent=extent,origin="lower",
               aspect=1,
               cmap="bwr",
               vmin = 0.1, vmax = 2.,
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

sub1.locator_params(nbins=5,axis='y')
sub1.locator_params(nbins=5,axis='x')

sub1.set_xlabel(r'$x\ [c/\omega_{pe}]$')
sub1.set_ylabel(r'$y\ [c/\omega_{pe}]$')

sub1.text(extent[1]/2+5,extent[1]+5,r"$n_e\ [(c/\omega_{pe})^{-3}]$")
