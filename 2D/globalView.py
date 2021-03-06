#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:06:38 2022

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
plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

st = slice(None,None,1)
time = o.getTimeAxis()[st]

mu = o.rqm[o.sIndex("iL")]
#----------------------------------------------

# kin_el = o.getEnergyIntegr(time, qty="kin", species="eL")
# kin_er = o.getEnergyIntegr(time, qty="kin", species="eR")
# kin_il = o.getEnergyIntegr(time, qty="kin", species="iL")
# kin_ir = o.getEnergyIntegr(time, qty="kin", species="iR")
av = (1,2)
Kin_UiL = np.mean(
                    (np.sqrt(
                        1+o.getUfluid(time, "iL", "x")**2+
                          o.getUfluid(time, "iL", "y")**2+
                          o.getUfluid(time, "iL", "z")**2)-1
                    )*mu,axis=av)

Kin_UeL = np.mean(
                    (np.sqrt(
                        1+o.getUfluid(time, "eL", "x")**2+
                          o.getUfluid(time, "eL", "y")**2+
                          o.getUfluid(time, "eL", "z")**2)-1
                    ),axis=av)


TiLx = np.mean(o.getUth(time, "iL", "x")**2,axis=av) * mu
# TiRx = np.mean(o.getUth(time, "iR", "x")**2,axis=(1,2)) * o.getRatioQM("iR")

TeLx = np.mean(o.getUth(time, "eL", "x")**2,axis=av)
# TeRx = np.mean(o.getUth(time, "eR", "x")**2,axis=(1,2))

TiLy = np.mean(o.getUth(time, "iL", "y")**2,axis=av) * mu
# TiRy = np.mean(o.getUth(time, "iR", "y")**2,axis=(1,2)) * o.getRatioQM("iR")

TeLy = np.mean(o.getUth(time, "eL", "y")**2,axis=av)
# TeRy = np.mean(o.getUth(time, "eR", "y")**2,axis=(1,2))


Ex,Ey,Ez = o.getEnergyIntegr(time, "E")
Bx,By,Bz = o.getEnergyIntegr(time, "B")
B = Bx+By+Bz


# GRavwB = np.gradient(normB)

# sat = []
# for i in range(len(GRavwB)-1):
#     if len(sat)<3 and GRavwB[i+1]<0 and GRavwB[i]>0: sat.append(i)

# sl_ew = slice(min_ew,max_ew)
# sl_iw = slice(min_iw,max_iw)
# amp_ew, index_ew, rsquared_ew = fit.fitExponential(time[sl_ew], avwB[sl_ew])
# amp_iw, index_iw, rsquared_iw = fit.fitExponential(time[sl_iw], avwB[sl_iw])

#%%
#----------------------------------------------
# fig, (sub1,sub2) = plt.subplots(1,2,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

# sub1.semilogy(time,kin_el,color="b",linestyle="--",label=r"$Kin_{eL}$")

# sub1.semilogy(time,kin_il,color="b",label=r"$Kin_{iL}$")
# sub1.semilogy(time,kin_ir,color="cyan",label=r"$Kin_{iR}$")
# for l in sat:
#     sub1.axvline(time[l],color="gray",linestyle="--",linewidth=0.7)
#     sub2.axvline(time[l],color="gray",linestyle="--",linewidth=0.7)

sub1.semilogy(time,B,color="g",label=r"$\mathcal{E}_B$")
# sub2.semilogy(time,normB,color="g",label=r"$B$")

# sub1.plot(time,viR,color="cyan")
sub1.semilogy(time,Ex,color="g",linestyle="--",label=r"$\mathcal{E}_{Ex}$")
sub1.semilogy(time,Ey,color="cyan",linestyle="--",label=r"$\mathcal{E}_{Ey}$")
# sub2.semilogy(time,normE,color="g",linestyle="--",label=r"$E$")

sub1.semilogy(time,Kin_UiL,color="b",label=r"$\mathcal{E}_{kin,iL|x}$")
# sub1.semilogy(time,np.abs(UiR),color="k",linestyle="--",label=r"$U_{iR}$")

sub1.semilogy(time,Kin_UeL,color="b",linestyle="--",label=r"$\mathcal{E}_{kin,eL|x}$")
# sub2.semilogy(time,np.abs(UeR),color="k",linestyle="--",label=r"$U_{eR}$")

sub1.semilogy(time,TiLx,color="orange",label=r"$T_{iL|x}$")
# sub2.semilogy(time,TiRx,color="orange",label=r"$T_{iRx}$")

sub1.semilogy(time,TeLx,color="orange",linestyle="--",label=r"$T_{eL|x}$")
# sub2.semilogy(time,TeRx,color="orange",linestyle="--",label=r"$T_{eRx}$")

sub1.semilogy(time,TiLy,color="r",label=r"$T_{iL|y}$")
# sub2.semilogy(time,TiRy,color="r",label=r"$T_{iRy}$")

sub1.semilogy(time,TeLy,color="r",linestyle="--",label=r"$T_{eL|y}$")
# sub2.semilogy(time,TeRy,color="r",linestyle="--",label=r"$T_{eRy}$")

sub1.set_xlim(time[0],time[-1])
sub1.set_ylim(1e-4,3e2)

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
# sub2.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")


# sub1.semilogy(time[sl_ew],amp_ew*np.exp(index_ew*time[sl_ew]),color="k")

# sub1.semilogy(time[sl_iw],amp_iw*np.exp(index_iw*time[sl_iw]),color="k")

# r = index_ew/index_iw
sub1.legend(frameon=False,ncol=4)
# sub2.legend(frameon=False)


plt.savefig(o.path+"/plots/globalView.png",dpi="figure")
