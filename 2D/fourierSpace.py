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

import parallelFunctions as pf
import fit

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,10)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis()[st]

plotSpectrum2D = True
findGamma = False

#----------------------------------------------
if plotSpectrum2D:
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
    axis_kX = np.fft.rfftfreq(len(x),x[1]-x[0]) *2*np.pi
    axis_kY = np.fft.rfftfreq(len(y),y[1]-y[0]) *2*np.pi

    eps = 0
    indMid = len(x)//2

    extent=(min(axis_kX),max(axis_kX),min(axis_kY),max(axis_kY))
    stages = pf.distrib_task(0, len(time)-1, o.nbrCores)

    #----------------------------------------------
    #----------------------------------------------
    data = o.getB(time,"z")

    ftB = np.abs(np.fft.rfft2(data))
    ftB = np.flip(ftB[:,:indMid+1,:],axis=-1) + eps

    #----------------------------------------------
    path = o.path+"/plots/fourierBz"
    o.setup_dir(path)

    it = ((ftB [s[0]:s[1]],
           time[s[0]:s[1]],
           extent, s[0], path) for s in stages)

    pf.parallel(plot2D, it, o.nbrCores, noInteract=True)
    """
    #----------------------------------------------
    #----------------------------------------------
    data = o.getE(time,"x")

    ftB = np.abs(np.fft.fft2(data,axes=(1,2)))
    ftB = ftB[:,:indMid+1,indMid-1:] + eps

    #----------------------------------------------
    path = o.path+"/plots/fourierEx"
    o.setup_dir(path)

    it = ((ftB [s[0]:s[1]],
           time[s[0]:s[1]],
           extent, s[0], path) for s in stages)

    pf.parallel(plot2D, it, o.nbrCores, noInteract=True)

    #----------------------------------------------
    #----------------------------------------------
    data = o.getE(time,"y")

    ftB = np.abs(np.fft.fft2(data,axes=(1,2)))
    ftB = ftB[:,:indMid+1,indMid-1:] + eps

    #----------------------------------------------
    path = o.path+"/plots/fourierEy"
    o.setup_dir(path)

    it = ((ftB [s[0]:s[1]],
           time[s[0]:s[1]],
           extent, s[0], path) for s in stages)

    pf.parallel(plot2D, it, o.nbrCores, noInteract=True)
    """


#----------------------------------------------
if findGamma:

    iFinal = 20
    time = time[:iFinal]
    ftB  = ftB[:iFinal]

    #----------------------------------------------
    def fit2D(ftB,axis_kX,axis_kY,time):

        rge_kY = range(len(axis_kY))
        gamma = np.zeros((len(axis_kX),len(axis_kY)))
        for kx in range(len(axis_kX)):

            print(kx)
            for ky in rge_kY:

                gamma[kx,ky] = fit.fitExponential(time, ftB[:,kx,ky])[1]

        return gamma


    #----------------------------------------------
    stages = pf.distrib_task(0, len(ftB[0])-1, o.nbrCores)

    it = ((ftB [:,s[0]:s[1]],
           axis_kX[s[0]:s[1]],
           axis_kY,
           time) for s in stages)

    gamma = pf.parallel(fit2D, it, o.nbrCores, noInteract=True)

    new_gamma = np.zeros((len(axis_kX),len(axis_kY)))
    c=0
    for i in range(len(gamma)):

        new_gamma[c:c+len(gamma[i])] = gamma[i]
        c+=len(gamma[i])
    gamma = new_gamma


    #----------------------------------------------
    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    im=sub1.imshow(gamma.T,   #no .T to keep first axis (time) along y
                    extent=extent,origin="lower",
                    aspect="auto",
                    cmap="jet",
                    norm=LogNorm(vmin=1e-2,vmax=1),
                    interpolation="None")

    divider = make_axes_locatable(sub1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    # sub1.locator_params(nbins=5,axis='y')
    # sub1.locator_params(nbins=5,axis='x')

    sub1.set_xlabel(r'$k_x\ [\omega_{pi}/c]$')
    sub1.set_ylabel(r'$k_y\ [\omega_{pi}/c]$')

    sub1.set_xlim(min(axis_kX),max(axis_kX))
    sub1.set_ylim(min(axis_kY),max(axis_kY))


#----------------------------------------------
"""
ftB = np.mean(
              np.abs(
                  np.fft.rfft(o.getB(time,"z"),axis=2)[:,:,1:]
                  ),
              axis=1)
# ftE = np.mean(fe,axis=1)
# std_ftB = np.std(fb,axis=1)
# std_ftE = np.std(fe,axis=1)

extent=(min(axis_kY),max(axis_kY),min(time),max(time))
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.set_xscale('log')
# sub1.set_yscale('log')

eps = 1e-7
im=sub1.imshow(ftB + eps,   #no .T to keep first axis (time) along y
               extent=extent,origin="lower",
                aspect="auto",
               cmap="jet",
                norm=LogNorm(vmin=1e-2,vmax=1e2),
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

# sub1.locator_params(nbins=5,axis='y')
# sub1.locator_params(nbins=5,axis='x')

sub1.set_xlabel(r'$k\ [\omega_{pi}/c]$')
sub1.set_ylabel(r'$t\ [\omega_{pi}^{-1}]$')

sub1.set_xlim(min(axis_kY),max(axis_kY))
"""

"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

for t in range(len(time)):
    sub1.loglog(axis_kY, ftB[t])
    # sub1.fill_between(axis_kY, ftB[t]-std_ftB[t],
    #                            ftB[t]+std_ftB[t],alpha=0.3)

sub1.set_xlabel(r"$k_y\ [\omega_{pi}/c]$")
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

for t in range(len(time)):
    sub1.loglog(axis_kY, ftE[t])
    # sub1.fill_between(axis_kY, ftE[t]-std_ftE[t],
    #                            ftE[t]+std_ftE[t],alpha=0.3)
"""
