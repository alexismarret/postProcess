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
run  ="CS3D"
o = osiris.Osiris(run,spNorm="iL")

x    = o.getAxis("x")
y    = o.getAxis("y")

time = o.getTimeAxis()
#----------------------------------------------

axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0])[1:] *2*np.pi
axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0])[1:] *2*np.pi

ftbz_perp = np.zeros((len(time),len(axis_kY)))
ftbz_para = np.zeros((len(time),len(axis_kX)))
for i in range(len(time)):
    B = o.getB(time[i], "z")
    ftbz_perp[i] = np.mean(
                   np.abs(np.fft.rfft(B,axis=1)[:,1:]),
                   axis=(0,2))
    ftbz_para[i] = np.mean(
                   np.abs(np.fft.rfft(B,axis=0)[1:]),
                   axis=(1,2))



#%%
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.set_xscale('log')
# sub1.set_yscale('log')

extent=(min(axis_kY),max(axis_kY),min(time),max(time))
eps=1e-7
im=sub1.imshow(ftbz_perp + eps,
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


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.set_xscale('log')
# sub1.set_yscale('log')

extent=(min(axis_kX),max(axis_kX),min(time),max(time))

eps=1e-7
im=sub1.imshow(ftbz_para + eps,
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

sub1.set_xlim(min(axis_kX),max(axis_kX))

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
