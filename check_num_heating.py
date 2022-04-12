#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:06:01 2022

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
        'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  = ("uniform2","uniform4","uniform8","uniform16")

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

for r in run:
    o = osiris.Osiris(r,spNorm="i")

    st = slice(None,None,1)
    time = o.getTimeAxis()[st]

    En_B = np.sum(o.getEnergyIntegr(time, "B"),axis=-1)
    En_E = np.sum(o.getEnergyIntegr(time, "E"),axis=-1)

    kin_e = o.getEnergyIntegr(time, qty="kin", species="e")
    kin_i = o.getEnergyIntegr(time, qty="kin", species="i")

    # U_int_i =  np.mean(o.getCharge(time,"i")*(o.getUth(time, "i", "x")**2 +
    #                                             o.getUth(time, "i", "y")**2 +
    #                                             o.getUth(time, "i", "z")**2),
    #                                             axis=1)/2*o.getRatioQM("i")

    # U_int_e = -np.mean(o.getCharge(time,"e")*(o.getUth(time, "e", "x")**2 +
    #                                             o.getUth(time, "e", "y")**2 +
    #                                             o.getUth(time, "e", "z")**2),
    #                                             axis=1)/2

    E_tot = En_E+En_B+kin_e+kin_i


    sub1.plot(time, (E_tot-E_tot[0])/E_tot[0], label=r"$%s$" %r)

sub1.legend(frameon=False)
sub1.set_xlim(time[0],time[-1])
sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$(\mathcal{E}-\mathcal{E}_0)/\mathcal{E}_0$")




