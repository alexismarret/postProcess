#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:10:43 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt
import fit

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS3D"
o = osiris.Osiris(run,spNorm="iL",nbrCores=6)

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
sl = (sx,sy,sz)

st = slice(1,None,1)
time = o.getTimeAxis()[st]

#----------------------------------------------
UiL   = o.getUfluid(time, "iL", "x", sl=sl,av=(1,2,3),parallel=False)

TiL = np.zeros(len(time))
TeL = np.zeros(len(time))
mu = o.getRatioQM("iL")
for i in range(len(time)):
    TiL[i] = np.mean(o.getUth   (time[i], "iL", "x", sl=sl,parallel=False)**2*mu,axis=(0,1,2))
    TeL[i] = np.mean(o.getUth   (time[i], "eL", "x", sl=sl,parallel=False)**2,   axis=(0,1,2))

#%%
#----------------------------------------------
# fig, (sub1,sub2) = plt.subplots(1,2,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

sub1.plot(time,UiL,color="g")
sub1.plot(time,TiL,color="r")
sub1.plot(time,TeL,color="b")

sub1.set_xlim(time[0],time[-1])
# sub1.set_ylim(1e-3,3e2)

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")

sub1.legend(frameon=False,ncol=4)



plt.savefig(o.path+"/plots/globalView.png",dpi="figure")
