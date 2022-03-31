#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:32:41 2022

@author: alexis
"""


import osiris
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run = "counterStream6"

o = osiris.Osiris(run,spNorm="iL")

#----------------------------------------------
timeFene = o.getTimeAxis(ene=True)
timePene = o.getTimeAxis(species="eL", ene=True)

En_E = o.getEnergyIntegr(timeFene, "E")
En_B = o.getEnergyIntegr(timeFene, "B")

kin_el = o.getEnergyIntegr(timePene, qty="kin", species="eL")
kin_il = o.getEnergyIntegr(timePene, qty="kin", species="iL")
kin_er = o.getEnergyIntegr(timePene, qty="kin", species="eR")
kin_ir = o.getEnergyIntegr(timePene, qty="kin", species="iR")

#----------------------------------------------
fig, ((sub1,sub2),(sub3,sub4)) = plt.subplots(2,2,figsize=(4.1,1.8),dpi=300)

sub1.semilogy(timePene,kin_el,label=r"$kin\ el$")
sub1.semilogy(timePene,kin_il,label=r"$kin\ il$")
sub2.semilogy(timePene,kin_er,label=r"$kin\ er$")
sub2.semilogy(timePene,kin_ir,label=r"$kin\ ir$")

sub3.semilogy(timeFene,En_E[:,0],label=r"$E_x$")
sub3.semilogy(timeFene,En_E[:,1],label=r"$E_y$")
sub3.semilogy(timeFene,En_E[:,2],label=r"$E_z$")

sub4.semilogy(timeFene,En_B[:,0],label=r"$B_x$")
sub4.semilogy(timeFene,En_B[:,1],label=r"$B_y$")
sub4.semilogy(timeFene,En_B[:,2],label=r"$B_z$")

sub1.legend(frameon=False)
sub2.legend(frameon=False)
sub3.legend(frameon=False)
sub4.legend(frameon=False)

plt.savefig(o.path+"/plots/integratedEnergyDensity.png",dpi="figure")


