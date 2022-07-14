#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:46:48 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run = "CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

ind = 512
sx = slice(None,ind,1)
sy = slice(None,ind,1)

sl = (sx,sy)
ax = (0,1)

st = slice(None,None,1)
time = o.getTimeAxis()[st]

#----------------------------------------------
workE_eL   = np.zeros(len(time))
workEc_eL  = np.zeros(len(time))
workEr_eL  = np.zeros(len(time))

B2 = o.getEnergyIntegr(time, qty="B")[2]
for i in range(len(time)):

    E  = o.getE(time[i], "x", sl=sl)
    Ec = o.getNewData(time[i], "Ecx", sl=sl)
    Er = E-Ec

    Ux_eL = o.getUfluid(time[i], "eL", "x", sl=sl)
    Uy_eL = o.getUfluid(time[i], "eL", "y", sl=sl)
    Uz_eL = o.getUfluid(time[i], "eL", "z", sl=sl)

    gamma_eL = np.sqrt(1+Ux_eL**2+Uy_eL**2+Uz_eL**2)
    workE_eL[i]  = -np.mean(Ux_eL * E / gamma_eL,axis=ax)
    workEc_eL[i] = -np.mean(Ux_eL * Ec / gamma_eL,axis=ax)
    workEr_eL[i] = -np.mean(Ux_eL * Er / gamma_eL,axis=ax)


#%%
#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,B2,color="k")

sub1.plot(time,workE_eL,label=r"$v_{ex}E_{x}$",color="r")
sub1.plot(time,workEc_eL,label=r"$v_{ex}E_{cx}$",color="b")
sub1.plot(time,workEr_eL,label=r"$v_{ex}E_{rx}$",color="g")

sub1.set_xlabel(r'$t\ [\omega_{pi}^{-1}]$')
sub1.set_ylabel(r'$v\cdot E\ [cE_0]$')

sub1.legend(frameon=False)
