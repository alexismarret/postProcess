#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:18:50 2022

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
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")


#----------------------------------------------
def plot2D(data,v1,v2,X,Y,time,extent,ind,figPath):

    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
    # fig.subplots_adjust(bottom=0.28)

    im=sub1.imshow(data[0,...].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="jet",
                   norm=LogNorm(vmin = 0.01, vmax = 0.1),
                   interpolation="None")

    # vecField = sub1.quiver(X,Y,v1[0,...],v2[0,...],color="k")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=5,axis='y')
    sub1.locator_params(nbins=5,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$n_i\ [(c/\omega_{pe})^{-3}]$",
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

        # vecField.remove()
        # vecField = sub1.quiver(X,Y,v1[i,...],v2[i,...],color="k",scale=8)

        plt.savefig(figPath+"/plot-{i}-time-{t}.png".format(i=i+ind,t=time[i]),dpi="figure")

    return

#----------------------------------------------
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

st = slice(None,None,1)
sx = slice(None,None,50)
sy = slice(None,None,50)
sl = (sx,sy)

x    = o.getAxis("x")
y    = o.getAxis("y")
time = o.getTimeAxis()[st]

X,Y = np.meshgrid(x[sx],y[sy])
UiLx = o.getUfluid(time, "iL","x",sl=sl)
UiLy = o.getUfluid(time, "iL","y",sl=sl)

#----------------------------------------------
eps=1e-7
# rI = o.getCharge(time, "iL") / (o.getCharge(time, "iR")+eps)
# niL = o.getCharge(time, "iL")+eps
TiL  = o.getUth(time, "iL", "x")**2*o.rqm[o.sIndex("iL")] + eps

#----------------------------------------------
stages = pf.distrib_task(0, len(time)-1, o.nbrCores)
extent=(min(x),max(x),min(y),max(y))

#----------------------------------------------
path = o.path+"/plots/TiL"
o.setup_dir(path)

it = ((TiL    [s[0]:s[1]],
       UiLx  [s[0]:s[1]],
       UiLy  [s[0]:s[1]],
       X,Y,
        time[s[0]:s[1]],
        extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

