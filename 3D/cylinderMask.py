#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:17:36 2022

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
plt.close("all")
#----------------------------------------------
run  ="CS3Drmhr"

o = osiris.Osiris(run,spNorm="iL")
species="iL"

st = slice(None)
time = o.getTimeAxis()[st]
t = 6

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (sx,sy,sz)

x = o.getAxis("x")[sx]
y = o.getAxis("y")[sy]
z = o.getAxis("z")[sz]
extent=(min(x),max(x),min(y),max(y))

#dimensions of cylinder
Lx = x[-1]-x[0]
Ly = y[-1]-y[0]
a = Lx/2
b = Ly/2
R = Lx/3

#True when outside cylinder
cond = ((x[:,None,None]-a)**2 + (y[None,:,None]-b)**2 > R**2) & (z[None,None,:]>=0)

plotFilter = False
plotMean = True
plotSlice = False
plotLine = False
checkSave = False
scale = 0.5

#----------------------------------------------
#filter data points inside cylinder
data = o.getCharge(time[t], species, sl=sl, parallel=False) / o.n0[0]
masked_data = np.ma.masked_where(cond, data, copy=False)

#----------------------------------------------
#perform average along line of sight (y direction)
mean_data = np.ma.mean(masked_data,axis=1)

#get density profile (mean over longitudinal x direction)
line_data = np.ma.mean(mean_data,axis=0)

#----------------------------------------------
path = o.path+"/plots/cylinder"
o.setup_dir(path, rm=False)

#----------------------------------------------
#dump data to txt file
savePath = path+"/xz_meanY-time-{t}.txt.gz".format(t=time[t])

#remove data not in cylinder (might be wrong by one cell, check!)
xl, xm = o.bisection(x,a-R), o.bisection(x,a+R)
keep = mean_data.data[xl+1:xm+1]

np.savetxt(savePath, keep)

#%%
#----------------------------------------------
if checkSave:
    loaded = np.loadtxt(savePath)

    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    im=sub1.imshow(loaded.T,
                   extent=(x[xl],x[xm],min(y),max(y)),origin="lower",
                   aspect=1,
                   cmap="coolwarm",
                   vmin = 1-scale, vmax = 1+scale)

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=10,axis='y')
    sub1.locator_params(nbins=10,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$z\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$<n_{iL}>_y\ [n_0]$",
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=sub1.transAxes)

    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[t],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)


#----------------------------------------------
if plotFilter:
    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    im=sub1.imshow(masked_data[...,0].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="coolwarm",
                   vmin = 1-scale, vmax = 1+scale)

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=10,axis='y')
    sub1.locator_params(nbins=10,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$n_{iL}\ [n_0]$",
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=sub1.transAxes)

    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[t],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    plt.savefig(path+"/xy_mask_cut-time-{t}.png".format(t=time[t]),dpi="figure")


#----------------------------------------------
if plotMean:
    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    im=sub1.imshow(mean_data.T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="coolwarm",
                   vmin = 1-scale, vmax = 1+scale)

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=10,axis='y')
    sub1.locator_params(nbins=10,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$z\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$<n_{iL}>_y\ [n_0]$",
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=sub1.transAxes)

    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[t],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    plt.savefig(path+"/xz_meanY-time-{t}.png".format(t=time[t]),dpi="figure")

#----------------------------------------------
if plotSlice:
    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    im=sub1.imshow(data[int(a),:,:].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="coolwarm",
                   vmin = 1-scale, vmax = 1+scale)

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=10,axis='y')
    sub1.locator_params(nbins=10,axis='x')

    sub1.set_xlabel(r'$y\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$z\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$n_{iL}\ [n_0]$",
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=sub1.transAxes)

    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[t],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    plt.savefig(path+"/yz_cut-time-{t}.png".format(t=time[t]),dpi="figure")

#----------------------------------------------
if plotLine:
    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    sub1.plot(y, line_data,color="g")

    sub1.set_xlabel(r'$z\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r"$<n_{iL}>_{x,y}$")

    sub1.set_ylim(0.7,1.3)

    txt = sub1.text(0.25, 1.04,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[t],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    plt.savefig(path+"/line_z-time-{t}.png".format(t=time[t]),dpi="figure")

    # fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    # sub1.loglog(kz, fft_line_data,color="b")

    # sub1.set_xlabel(r'$k_z\ [\omega_{pi}/c]$')
    # sub1.set_ylabel(r"$FT[<n_{iL}>_{x,y}]$")

