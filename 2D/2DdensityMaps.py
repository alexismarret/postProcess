#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:00:25 2022

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
        'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
def plot2D(data,time,extent,ind,figPath):

    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
    # fig.subplots_adjust(bottom=0.28)

    im=sub1.imshow(data[0,...].T,
                   extent=extent,origin="lower",
                    aspect=1,
                   cmap="bwr",
                    norm=LogNorm(vmin = 0.1, vmax = 10),
                   interpolation="None")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=20,axis='y')
    sub1.locator_params(nbins=20,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$n_{iL}/n_0$",
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
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(0,None,1)
sy = slice(0,None,1)
st = slice(None,None,1)
sl=(sx,sy)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sy]
time = o.getTimeAxis()[st]

species="eL"
pChargeSign = o.rqm[o.sIndex(species)]/np.abs(o.rqm[o.sIndex(species)])
n0 = o.n0[o.sIndex(species)]

#----------------------------------------------
eps = 1e-7   #avoid 0
niL = o.getCharge(time, species,sl=sl) / n0 * pChargeSign


#----------------------------------------------
stages = pf.distrib_task(0, len(time)-1, o.nbrCores)
extent=(min(x),max(x),min(y),max(y))

#----------------------------------------------
path = o.path+"/plots/neL"
o.setup_dir(path)

it = ((niL [s[0]:s[1]],
        time[s[0]:s[1]],
        extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

"""
#----------------------------------------------
eps = 1e-6   #avoid /0
ratio = o.getCharge(time, "iL",sl=sl) / o.getCharge(time, "iR",sl=sl)+eps


#----------------------------------------------
stages = pf.distrib_task(0, len(time)-1, o.nbrCores)
extent=(min(x),max(x),min(y),max(y))

#----------------------------------------------
path = o.path+"/plots/rNi"
o.setup_dir(path)

it = ((ratio [s[0]:s[1]],
        time[s[0]:s[1]],
        extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

"""
