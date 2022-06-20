#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:20:59 2022

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
import fit

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------

def plot2D(data,time,extent,ind,figPath):

    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    fig.subplots_adjust(bottom=0.15)
    # fig.subplots_adjust(left=0.15)

    im=sub1.imshow(data[0,...].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="jet",
                   norm=LogNorm(vmin=1e-2,vmax=1e2),
                    interpolation="None")


    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, ax=sub1)

    sub1.locator_params(nbins=10,axis='y')
    sub1.locator_params(nbins=10,axis='x')

    sub1.set_xlabel(r'$k_x\ [\omega_{pi}/c]$')
    sub1.set_ylabel(r'$k_y\ [\omega_{pi}/c]$')

    # sub1.text(1, 1.05,
    #           r"$n_{iL}/n_0$",
    #           horizontalalignment='right',
    #           verticalalignment='bottom',
    #           transform=sub1.transAxes)

    txt = sub1.text(0.35, 1.02,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[0],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    #needed to avoid change of figsize
    plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=0+ind,t=time[0]),dpi="figure")

    for i in range(len(time)):

        Artist.remove(txt)
        txt = sub1.text(0.35, 1.02,
                        r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[i],
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=sub1.transAxes)

        im.set_array(data[i,...].T)

        plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=i+ind,t=time[i]),dpi="figure")

    return

#----------------------------------------------
run  ="CS3Dtrack"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (sx,sy,0)

x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sy]
z    = o.getAxis("y")[sz]

st = slice(None,None,1)
time = o.getTimeAxis()[st]

#----------------------------------------------
axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0]) *2*np.pi
axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0]) *2*np.pi

eps = 0
indMid = len(x)//2

extent=(min(axis_kX),max(axis_kX),min(axis_kY),max(axis_kY))
stages = pf.distrib_task(0, len(time)-1, o.nbrCores)

#----------------------------------------------
#----------------------------------------------
data = o.getB(time,"x",sl=sl,parallel=False)

ftB = np.abs(np.fft.rfft2(data))
ftB = np.flip(ftB[:,:indMid+1,:],axis=-1) + eps

#----------------------------------------------
path = o.path+"/plots/fourierBx"
o.setup_dir(path)

it = ((ftB [s[0]:s[1]],
       time[s[0]:s[1]],
       extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

#----------------------------------------------
#----------------------------------------------
data = o.getB(time,"y",sl=sl,parallel=False)

ftB = np.abs(np.fft.rfft2(data))
ftB = np.flip(ftB[:,:indMid+1,:],axis=-1) + eps

#----------------------------------------------
path = o.path+"/plots/fourierBy"
o.setup_dir(path)

it = ((ftB [s[0]:s[1]],
       time[s[0]:s[1]],
       extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

#----------------------------------------------
#----------------------------------------------
data = o.getE(time,"x",sl=sl,parallel=False)

ftB = np.abs(np.fft.rfft2(data))
ftB = np.flip(ftB[:,:indMid+1,:],axis=-1) + eps

#----------------------------------------------
path = o.path+"/plots/fourierEx"
o.setup_dir(path)

it = ((ftB [s[0]:s[1]],
       time[s[0]:s[1]],
       extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

#----------------------------------------------
#----------------------------------------------
data = o.getE(time,"y",sl=sl,parallel=False)

ftB = np.abs(np.fft.rfft2(data))
ftB = np.flip(ftB[:,:indMid+1,:],axis=-1) + eps

#----------------------------------------------
path = o.path+"/plots/fourierEy"
o.setup_dir(path)

it = ((ftB [s[0]:s[1]],
       time[s[0]:s[1]],
       extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

