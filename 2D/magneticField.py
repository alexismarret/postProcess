#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:18:07 2022

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
# plt.close("all")

#----------------------------------------------
run  ="counterStreamFast"
o = osiris.Osiris(run,spNorm="iL")

st = slice(None,None,1)
time = o.getTimeAxis()[st]

#----------------------------------------------
En_B = o.getEnergyIntegr(time, "B")
En_E = o.getEnergyIntegr(time, "E")

En_Bx, En_By, En_Bz = En_B[...,0], En_B[...,1], En_B[...,2]
En_Ex, En_Ey, En_Ez = En_E[...,0], En_E[...,1], En_E[...,2]



#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,1.8),dpi=300)

sub1.plot(time,En_Bx,label=r"$\mathcal{E}_{Bx}$")
sub1.plot(time,En_By,label=r"$\mathcal{E}_{By}$")
sub1.plot(time,En_Bz,label=r"$\mathcal{E}_{Bz}$")

sub1.plot(time,En_Ex,label=r"$\mathcal{E}_{Ex}$")
sub1.plot(time,En_Ey,label=r"$\mathcal{E}_{Ey}$")
sub1.plot(time,En_Ez,label=r"$\mathcal{E}_{Ez}$")

sub1.legend(frameon=False)
sub1.set_xlim(time[0],time[-1])
sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
# sub1.set_ylabel(r"$(\mathcal{E}-\mathcal{E}_0)/\mathcal{E}_0$")
