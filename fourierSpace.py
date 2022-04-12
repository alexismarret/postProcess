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

import parallel_functions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
def plot2D(data,time,extent,ind,figPath):

    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
    # fig.subplots_adjust(bottom=0.09)

    sub1.set_xscale('log')
    sub1.set_yscale('log')

    im=sub1.imshow(data[0,...].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="hot",norm=LogNorm(vmin = 1e-1, vmax = 1e2),
                   interpolation="None")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    # sub1.locator_params(nbins=5,axis='y')
    # sub1.locator_params(nbins=5,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$T\ [m_ec^2]$",
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=sub1.transAxes)

    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[0],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    #needed to avoid change of figsize
    plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=0+ind,t=time[0]),dpi="figure")

    for i in range(len(time)):

        Artist.remove(txt)
        txt = sub1.text(0.35, 1.05,
                        r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[i],
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=sub1.transAxes)

        im.set_array(data[i,...].T)

        plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=i+ind,t=time[i]),dpi="figure")

    return

#----------------------------------------------
run  ="counterStream"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,2)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis()[st]

#----------------------------------------------
normB = np.sqrt(o.getB(time,"x")**2+
                o.getB(time,"y")**2+
                o.getB(time,"z")**2)

normE = np.sqrt(o.getE(time,"x")**2+
                o.getE(time,"y")**2+
                o.getE(time,"z")**2)

axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0])[1:] *2*np.pi
axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0])[1:] *2*np.pi


ftB = np.zeros((len(time),len(x),len(y)))
ftE = np.zeros((len(time),len(x),len(y)))

for i in range(len(time)):
    ftB[i] = np.abs(np.fft.fft2(normB[i]))
    ftE[i] = np.abs(np.fft.fft2(normE[i]))

ftB = ftB[:,len(x)//2:,len(y)//2:]
ftE = ftE[:,len(x)//2:,len(y)//2:]

stages = pf.distrib_task(0, len(time)-1, o.nbrCores)
extent=(min(axis_kX),max(axis_kX),min(axis_kY),max(axis_kY))


#----------------------------------------------
path = o.path+"/plots/fftB"
o.setup_dir(path)

it = ((ftB  [s[0]:s[1]],
        time        [s[0]:s[1]],
        extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, plot=True)

#----------------------------------------------
path = o.path+"/plots/fftE"
o.setup_dir(path)

it = ((ftE  [s[0]:s[1]],
        time        [s[0]:s[1]],
        extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, plot=True)
