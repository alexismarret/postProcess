#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:28:16 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)

#----------------------------------------------
# run = 'CS3DMu64Ma20theta80'
run = "CS3Dtrack"

o = osiris.Osiris(run,spNorm="iL")

ind = 50
sx = slice(None,ind,1)
sy = slice(None,ind,1)
sz = slice(None,ind,1)
sl = (sx,sy,sz)

st = slice(0,-1,1)
time = o.getTimeAxis()[st]
dt = time[1]-time[0]

avnp = (0,1,2)
mass = o.rqm[o.sIndex("iL")]
lorentz0 = np.sqrt(1+np.sum(o.ufl[0]**2))
v0 = o.ufl[0,0]/lorentz0
eps0 = lorentz0-1

#----------------------------------------------
Te  = np.zeros(len(time))
Ex = np.zeros(len(time))
Ecx = np.zeros(len(time))
Erx = np.zeros(len(time))

for i in range(len(time)):
    Te[i]  = np.mean(o.getUth(time[i], "eL", "x", sl=sl)**2, axis=avnp)
    E = o.getE(time[i], "x", sl=sl)
    Ex[i] = np.mean(E**2, axis=avnp)/2
    Ecx[i] = np.mean(o.getNewData(time[i], "Ecx", sl=sl)**2, axis=avnp)/2
    Erx[i] = np.mean((E - Ecx[i])**2, axis=avnp)/2


# Ex,Ey,Ez = o.getEnergyIntegr(time, "E")
# Bx,By,Bz = o.getEnergyIntegr(time, "B")

# E2 = Ex+Ey+Ez
# B2 = Bx+By+Bz

#%%
# plt.close("all")
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,Te*o.n0[0] ,color="b")
# sub1.plot(time,B2*(1-v0**2),color="g")
# sub1.plot(time,E2,color="r")

factor = 1/v0**2-1

sub1.plot(time,Ex,color="r")
sub1.plot(time,Ecx,color="k")
sub1.plot(time,Erx,color="k",linestyle="--")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
