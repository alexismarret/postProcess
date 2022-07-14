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
plt.close("all")

#----------------------------------------------
run = 'CS3DMu64Ma20theta80'
# run = "CS3Dtrack"

o = osiris.Osiris(run,spNorm="iL",globReduced=True)

ind = 100
sx = slice(None,ind,1)
sy = slice(None,ind,1)
sz = slice(None,ind,1)
sl = (sx,sy,sz)

st = slice(0,-1,1)
time = o.getTimeAxis()[st]
dt = time[1]-time[0]

avnp = (0,1,2)
mass = o.rqm[o.sIndex("iL")]

#----------------------------------------------
Te = np.zeros(len(time))

for i in range(len(time)):
    Te[i] = np.mean(o.getUth(time[i], "eL", "x", sl=sl)**2, axis=avnp)

Ex,Ey,Ez = o.getEnergyIntegr(time, "E")
Bx,By,Bz = o.getEnergyIntegr(time, "B")

E2 = Ex+Ey+Ez
B2 = Bx+By+Bz

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,Te,color="b")
sub1.plot(time,B2,color="g")
sub1.plot(time,E2*64,color="r")


