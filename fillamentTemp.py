#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:34:26 2022

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
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1.5,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="counterStream"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,1)
x     = o.getAxis("x")[sx]
y     = o.getAxis("y")[sx]
time = o.getTimeAxis("eL")[st]

#----------------------------------------------
mask_pos = o.locFilament(time, polarity=+1, fac=2)
mask_neg = o.locFilament(time, polarity=-1, fac=2)
mask_nf  = ~(mask_pos*mask_neg)
TiLx = o.getUth(time, "iL", "x")
TiLy = o.getUth(time, "iL", "y")

# #f: value in filament
# #nf: value out of filament
TiLx_f_pos  = np.ma.mean(np.ma.masked_array(TiLx,mask= mask_pos),axis=(1,2))
TiLx_f_neg  = np.ma.mean(np.ma.masked_array(TiLx,mask= mask_neg),axis=(1,2))

TiLy_f_pos  = np.ma.mean(np.ma.masked_array(TiLy,mask= mask_pos),axis=(1,2))
TiLy_f_neg  = np.ma.mean(np.ma.masked_array(TiLy,mask= mask_neg),axis=(1,2))

TiLx_nf =     np.ma.mean(np.ma.masked_array(TiLx,mask= mask_nf),axis=(1,2))
TiLy_nf =     np.ma.mean(np.ma.masked_array(TiLy,mask= mask_nf),axis=(1,2))

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

# sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(time,TiLx_f_pos,color="r",label=r"$T_{ix}^*/T_{ix}$")
sub1.plot(time,TiLy_f_pos,linestyle="--",color="r",label=r"$T_{iy}^*/T_{iy}$")

sub1.plot(time,TiLx_f_neg,color="b",label=r"$T_{ix}^*/T_{ix}$")
sub1.plot(time,TiLy_f_neg,linestyle="--",color="b",label=r"$T_{iy}^*/T_{iy}$")

sub1.plot(time,TiLx_nf,color="k",label=r"$T_{ix}^*/T_{ix}$")
sub1.plot(time,TiLy_nf,linestyle="--",color="k",label=r"$T_{iy}^*/T_{iy}$")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_xlim(min(time),max(time))

sub1.legend(frameon=False)








# j = (o.getCurrent(time, "eL", "x")+
#      o.getCurrent(time, "eR", "x")+
#      o.getCurrent(time, "iL", "x")+
#      o.getCurrent(time, "iR", "x"))

# j_pos = np.ma.masked_array(j,mask= mask_pos,copy=False)
# j_neg = np.ma.masked_array(j,mask= mask_neg,copy=False)
# j_nf  = np.ma.masked_array(j,mask= mask_nf, copy=False)

# t = 10
# extent=(min(x),max(x),min(y),max(y))

# def plot2D(data,time,extent):
#     #----------------------------------------------
#     fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
#     # fig.subplots_adjust(bottom=0.28)

#     im=sub1.imshow(data.T,
#                    extent=extent,origin="lower",
#                    aspect=1,
#                    cmap="bwr",
#                    vmin=-0.05,vmax=0.05,
#                    interpolation="None")

#     divider = make_axes_locatable(sub1)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     fig.colorbar(im, cax=cax)

#     sub1.locator_params(nbins=5,axis='y')
#     sub1.locator_params(nbins=5,axis='x')

#     sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
#     sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

#     sub1.text(1, 1.05,
#               r"$U_{eL}\ [c]$",
#               horizontalalignment='right',
#               verticalalignment='bottom',
#               transform=sub1.transAxes)

#     sub1.text(0.35, 1.05,
#               r"$t=%.1f\ [\omega_{pe}^{-1}]$"%time,
#               horizontalalignment='right',
#               verticalalignment='bottom',
#               transform=sub1.transAxes)

#     return


# plot2D(j[t],time[t],extent)

# plot2D(j_pos[t],time[t],extent)

# plot2D(j_neg[t],time[t],extent)

# plot2D(j_nf[t],time[t],extent)




"""
#----------------------------------------------
eps=1e-5
r = o.getUth(time, "iL", "x") / (o.getUth(time, "iR", "x")+eps)

mask = np.ma.getmask(np.ma.masked_where(r<1,r,copy=False))


r_g1 = np.ma.mean(np.ma.masked_array(r,mask= mask,copy=False),axis=(1,2))
r_l1 = np.ma.mean(np.ma.masked_array(r,mask=~mask,copy=False),axis=(1,2))

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)

sub1.plot(time,r_g1,color="r",label=r"$\langle T_{iL}/T_{iR}>1\rangle$")
sub1.plot(time,r_l1,color="b",label=r"$\langle T_{iL}/T_{iR}<1\rangle$")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_xlim(min(time),max(time))

sub1.legend(frameon=False)
"""



