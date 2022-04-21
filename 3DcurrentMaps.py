#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:10:53 2022

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

    im=sub1.imshow(data[0,...].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="bwr",
                   vmin = -0.1, vmax = 0.1,
                   interpolation="None")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=5,axis='y')
    sub1.locator_params(nbins=5,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$J\ [en_ec]$",
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
run  ="CS3D"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,1)
x     = o.getAxis("x")[sx]
y     = o.getAxis("y")[sx]
time = o.getTimeAxis("iL")[st]

#----------------------------------------------
# jiLx = o.getCurrent(time, "iL", "x")
jTotX = (o.getCurrent(time, "eL", "x")+
         o.getCurrent(time, "eR", "x")+
         o.getCurrent(time, "iL", "x")+
         o.getCurrent(time, "iR", "x"))


#----------------------------------------------
stages = pf.distrib_task(0, len(time)-1, o.nbrCores)
extent=(min(x),max(x),min(y),max(y))


#----------------------------------------------
path = o.path+"/plots/jTotX"
o.setup_dir(path)

it = ((jTotX  [s[0]:s[1]],
       time        [s[0]:s[1]],
       extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, plot=True)




mask = o.locFilament(time)

#----------------------------------------------
path = o.path+"/plots/jTotX_mask"
o.setup_dir(path)

jTotX_masked = np.ma.masked_array(jTotX,mask= mask,copy=False)

it = ((jTotX_masked  [s[0]:s[1]],
       time        [s[0]:s[1]],
       extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, plot=True)