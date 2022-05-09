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

av = (0,1,2)
mu = o.rqm[o.sIndex("iL")]

#----------------------------------------------
#ekin = (gamma-1)mpc**2
Ekin_iL = np.zeros(len(time))
TiLx = np.zeros(len(time))
TeLx = np.zeros(len(time))
TiLy = np.zeros(len(time))
TeLy = np.zeros(len(time))

for i in range(len(time)):
    Ekin_iL[i]  = np.mean(
                        (np.sqrt(
                            1+o.getUfluid(time[i], "iL", "x", sl=sl)**2+
                              o.getUfluid(time[i], "iL", "y", sl=sl)**2+
                              o.getUfluid(time[i], "iL", "z", sl=sl)**2)-1
                        )*mu,axis=av)

    TiLx[i] = np.mean(o.getUth(time[i], "iL", "x", sl=sl)**2*mu,axis=av)
    TeLx[i] = np.mean(o.getUth(time[i], "eL", "x", sl=sl)**2,   axis=av)
    TiLy[i] = np.mean(o.getUth(time[i], "iL", "y", sl=sl)**2*mu,axis=av)
    TeLy[i] = np.mean(o.getUth(time[i], "eL", "y", sl=sl)**2,   axis=av)

E = o.getEnergyIntegr(time, "E")
Ex = E[...,0]
Ey = E[...,1]
Ez = E[...,2]
B = o.getEnergyIntegr(time, "B")
Bx = B[...,0]
By = B[...,1]
Bz = B[...,2]

#%%
#----------------------------------------------
# fig, (sub1,sub2) = plt.subplots(1,2,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

sub1.plot(time,Ekin_iL,color="g",label=r"$\mathcal{E}_{kin|iL}$")
sub1.plot(time,TiLx,color="r",label=r"$T_{x|iL}$")
sub1.plot(time,TeLx,color="b",label=r"$T_{x|eL}$")

sub1.plot(time,TiLy,color="r",label=r"$T_{y|iL}$",linestyle="dashed")
sub1.plot(time,TeLy,color="b",label=r"$T_{y|eL}$",linestyle="dashed")

sub1.plot(time,Ex,color="k",linestyle="dotted",label=r"$E_x^2/2$")
sub1.plot(time,Ey,color="k",linestyle="dashed",label=r"$E_y^2/2$")
sub1.plot(time,Ez,color="k",linestyle="dashdot",label=r"$E_z^2/2$")

sub1.plot(time,Bx,color="gray",linestyle="dotted",label=r"$B_x^2/2$")
sub1.plot(time,By,color="gray",linestyle="dashed",label=r"$B_y^2/2$")
sub1.plot(time,Bz,color="gray",linestyle="dashdot",label=r"$B_z^2/2$")

sub1.set_xlim(time[0],time[-1])
# sub1.set_ylim(1e-3,3e2)

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")

sub1.legend(frameon=False,ncol=4)



plt.savefig(o.path+"/plots/globalView.png",dpi="figure")
