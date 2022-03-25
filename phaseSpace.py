#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:34:15 2022

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
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")


#----------------------------------------------
def plot2D(data,time,extent,ind,figPath):

    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
    # fig.subplots_adjust(bottom=0.09)

    im=sub1.imshow(data[0,...],
                   extent=extent,origin="lower",
                   aspect="auto",
                   cmap="hot",
                   vmin=-200,vmax=0,
                   interpolation="None")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=5,axis='y')
    sub1.locator_params(nbins=5,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pe}]$')
    sub1.set_ylabel(r'$p\ []$')

    # sub1.text(extent[1]/2+5,extent[1]+5,r"$u\ [c]$")
    # txt = sub1.text(extent[0],extent[1]+5,r"$t=%.1f\ [\omega_{pe}^{-1}]$"%time[0])

    #needed to avoid change of figsize
    plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=0+ind,t=time[0]),dpi="figure")

    for i in range(len(time)):

        # Artist.remove(txt)
        # txt = sub1.text(extent[0],extent[1]+5,r"$t=%.1f\ [\omega_{pe}^{-1}]$"%time[i])

        im.set_array(data[i,...])

        plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=i+ind,t=time[i]),dpi="figure")

    return

#----------------------------------------------
run  ="counterStream3"
o = osiris.Osiris(run,nbrCores=6)

sx  = slice(None,None,1)
stp = slice(None,None,1)
x      = o.getAxis("x")[sx]
y      = o.getAxis("y")[sx]
timeP  = o.getTimeParticles("eL")[stp]

#----------------------------------------------
p1x1 = o.getPhaseSpace(timeP, "eL", direction="x", comp="x")[...,sx]

#----------------------------------------------
stages = pf.distrib_task(0, len(timeP)-1, o.nbrCores)
extent=(0,len(p1x1[0,0]),0,256)

#----------------------------------------------
path = o.path+"/plots/p1x1"
o.setup_dir(path)

it = ((p1x1  [s[0]:s[1]].T,
       timeP [s[0]:s[1]],
       extent, s[0], path) for s in stages)

# pf.parallel(plot2D, it, o.nbrCores, plot=True)

a = p1x1[0,...,0]



fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(a)
