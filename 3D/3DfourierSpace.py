#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:19:44 2022

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
        'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)

#----------------------------------------------
run  ="CS3Dtrack"
o = osiris.Osiris(run,spNorm="iL")

x    = o.getAxis("x")
y    = o.getAxis("y")

time = o.getTimeAxis()
eps=1e-7
#----------------------------------------------

axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0])[1:] *2*np.pi
axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0])[1:] *2*np.pi

ft_perp = np.zeros((len(time),len(axis_kY)))
ft_para = np.zeros((len(time),len(axis_kX)))

for i in range(len(time)):

    data = o.getNewData(time[i], "Ecx")
    ft_perp[i] = np.mean(
                    np.abs(np.fft.rfft(data,axis=1))[:,1:],
                    axis=(0,2))
    ft_para[i] = np.mean(
                    np.abs(np.fft.rfft(data,axis=0))[1:],
                    axis=(1,2))


#%%
plt.close("all")
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

extent = o.imExtent(axis_kY, time)

im=sub1.imshow(ft_perp + eps,
               extent=extent,origin="lower",
                aspect="auto",
               cmap="jet",
                norm=LogNorm(vmin=1e-2,vmax=1e2),
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

sub1.set_yticks(time[::2])
sub1.set_xscale('log')

sub1.set_xlabel(r'$k_y\ [\omega_{pi}/c]$')
sub1.set_ylabel(r'$t\ [\omega_{pi}^{-1}]$')
sub1.grid(axis = 'both',color="silver",linewidth=0.4)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

extent = o.imExtent(axis_kX, time)

im=sub1.imshow(ft_para + eps,
               extent=extent,origin="lower",
                aspect="auto",
               cmap="jet",
                norm=LogNorm(vmin=1e-2,vmax=1e2),
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

sub1.set_yticks(time[::2])
sub1.set_xscale('log')

sub1.set_xlabel(r'$k_x\ [\omega_{pi}/c]$')
sub1.set_ylabel(r'$t\ [\omega_{pi}^{-1}]$')
sub1.grid(axis = 'both',color="silver",linewidth=0.4)


"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)


sub1.loglog(axis_kY, ftBpara)
sub1.fill_between(axis_kY, ftBpara-std_ftBpara,
                           ftBpara+std_ftBpara,alpha=0.3)

sub1.set_xlabel(r"$k_y\ [\omega_{pi}/c]$")
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)


sub1.loglog(axis_kY, ftBperp)
sub1.fill_between(axis_kY, ftBperp-std_ftBperp,
                           ftBperp+std_ftBperp,alpha=0.3)
"""
