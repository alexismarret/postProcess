#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:24:39 2022

@author: alexis
"""


#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt
import parallelFunctions as pf
from matplotlib.colors import LogNorm
from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
# run  ="CS2DrmhrTrack"
run = "CS2DrmhrRawLall"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
sl=(sx,sy)
st = slice(None,None,1)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis()[st]

#----------------------------------------------
def cor(a,b):

    #naive implementation, might have issue because of machine precision
    # c = np.mean((a - np.mean(a,axis=(1,2))[...,None,None]) *
    #             (b - np.mean(b,axis=(1,2))[...,None,None]),
    #             axis=(1,2))

    #shifted implementation, more robust
    v = 1  #grid point to use, arbitrary value
    kx = a[:,v,v][:,None,None]
    ky = b[:,v,v][:,None,None]

    ax=(1,2)
    c = ((np.nanmean((a-kx) * (b-ky),axis=ax) -
         np.nanmean(a-kx,axis=ax) * np.nanmean(b-ky,axis=ax)) /
         (np.nanstd((a),axis=ax)*np.nanstd((b),axis=ax)))

    return c

#----------------------------------------------
def plot2D(data,time,extent,ind,figPath):

    fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
    # fig.subplots_adjust(bottom=0.09)

    im=sub1.imshow(data[0,...].T,
                   extent=extent,origin="lower",
                   aspect=1,
                   cmap="jet",
                    # norm=LogNorm(vmin = 1e0, vmax = 1e1),
                    vmin = 0, vmax = 5,
                   interpolation="None")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    sub1.locator_params(nbins=20,axis='y')
    sub1.locator_params(nbins=20,axis='x')

    sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
    sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

    sub1.text(1, 1.05,
              r"$Skew$",
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
eps = 1e-8
skew = o.getNewData(time, "skew",sl=sl)
niL = o.getCharge(time, "iL",sl=sl)
TiL = o.getUth(time, "iL", "x",sl=sl) **2*o.rqm[o.sIndex("iL")]

cor_sT    = cor(skew, TiL)
cor_sn    = cor(skew, niL)

extent=(min(x),max(x),min(y),max(y))
stages = pf.distrib_task(0, len(time)-1, o.nbrCores)

"""
#----------------------------------------------
path = o.path+"/plots/skewness"
o.setup_dir(path)

it = ((skew  [s[0]:s[1]],
        time        [s[0]:s[1]],
        extent, s[0], path) for s in stages)

pf.parallel(plot2D, it, o.nbrCores, noInteract=True)
"""

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(0,linestyle="--",color="gray",linewidth=0.7)
sub1.plot(time,cor_sT,color="k",label=r"$Cor(skew(x,y),T(x,y))$")
sub1.plot(time,cor_sn,color="orange",label=r"$Cor(skew(x,y),n(x,y))$")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")

# sub1.set_ylim(-0.0015,0.0015)
sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.legend(frameon=False)
