#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:08:31 2022

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
# plt.close("all")

#----------------------------------------------
run  ="CS3Drmhr"
o = osiris.Osiris(run,spNorm="iL")
species ="iL"

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (0,sy,sz)

x     = o.getAxis("x")[sx]
y     = o.getAxis("y")[sy]
z     = o.getAxis("z")[sz]
extent=(min(x),max(x),min(y),max(y))

st = slice(None)
time = o.getTimeAxis()[st]
mu = o.rqm[o.sIndex(species)]

#----------------------------------------------
path = o.path+"/plots/niL"
o.setup_dir(path)

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

data = o.getCharge(time[0], species,sl=sl)

im=sub1.imshow(data.T,
               extent=extent,origin="lower",
               aspect=1,
               cmap="bwr",
               norm=LogNorm(vmin=1e-1,vmax=1e1),
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

sub1.locator_params(nbins=5,axis='y')
sub1.locator_params(nbins=5,axis='x')

sub1.set_xlabel(r'$y\ [c/\omega_{pi}]$')
sub1.set_ylabel(r'$z\ [c/\omega_{pi}]$')

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
plt.savefig(path+"/plot-{i}-time-{t}.png".format(i=0,t=time[0]),dpi="figure")

for i in range(len(time)):

    Artist.remove(txt)
    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[i],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    data = o.getCharge(time[i], species,sl=sl)
    im.set_array(data.T)

    plt.savefig(path+"/plot-{i}-time-{t}.png".format(i=i,t=time[i]),dpi="figure")

#----------------------------------------------
plt.close()
