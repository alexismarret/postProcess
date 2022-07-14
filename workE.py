#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:56:55 2022

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
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run ="CS3Dtrack"
# run="testTrackSingle"

spNorm = "iL"
o = osiris.Osiris(run,spNorm=spNorm)

species ="eL"

#----------------------------------------------
#index of macroparticle
tmax = None
sPart = slice(None,None,1)
sTime = slice(None,tmax,1)
sl = (sPart,sTime)

nM = (slice(None),slice(None,-1,None))    # n-1/2 and n
nP = (slice(None),slice(1,None,None))   # n+1/2

#----------------------------------------------
mass   = np.abs (o.rqm[o.sIndex(species)])
charge = np.sign(o.rqm[o.sIndex(species)])
dt = o.dt

#macroparticle, iteration
ene = o.getTrackData(species, "ene",sl=sl) * mass
t   = o.getTrackData(species, "t",  sl=(0,sTime)) / np.sqrt(np.abs(o.rqm[o.sIndex(spNorm)]))
p1  = o.getTrackData(species, "p1", sl=sl)
p2  = o.getTrackData(species, "p2", sl=sl)
p3  = o.getTrackData(species, "p3", sl=sl)
e1  = o.getTrackData(species, "E1", sl=sl)
e2  = o.getTrackData(species, "E2", sl=sl)
e3  = o.getTrackData(species, "E3", sl=sl)

#----------------------------------------------
def lorentz(p1,p2,p3):

    return np.sqrt(1.+p1**2+p2**2+p3**2)

#----------------------------------------------
lorentz1 = lorentz(p1[nM], p2[nM], p3[nM])
lorentz2 = lorentz(p1[nP], p2[nP], p3[nP])

workE_centered_X = charge/2 * (p1[nM]/lorentz1 + p1[nP]/lorentz2) * e1[nM]
workE_centered_Y = charge/2 * (p2[nM]/lorentz1 + p2[nP]/lorentz2) * e2[nM]
workE_centered_Z = charge/2 * (p3[nM]/lorentz1 + p3[nP]/lorentz2) * e3[nM]

#----------------------------------------------
I_workE_centered_X = np.cumsum(workE_centered_X,axis=1)*dt + ene[:,0][...,None]
I_workE_centered_Y = np.cumsum(workE_centered_Y,axis=1)*dt + ene[:,0][...,None]
I_workE_centered_Z = np.cumsum(workE_centered_Z,axis=1)*dt + ene[:,0][...,None]

I_workE_centered   = np.cumsum(workE_centered_X +
                               workE_centered_Y +
                               workE_centered_Z, axis=1)*dt + ene[:,0][...,None]

Err = np.abs(I_workE_centered - ene[nP]) / ene[nP]

# #----------------------------------------------
# time = o.getTimeAxis()
# avnp = (0,1,2)
# sl=(slice(0,int(o.grid[0]/2),1),
#     slice(0,int(o.grid[1]/2),1),
#     slice(0,int(o.grid[2]/2),1))

# TeLx = np.zeros(len(time))
# TeLy = np.zeros(len(time))

# for i in range(len(time)):
#     print(i)
#     TeLx[i] = np.mean(o.getUth(time[i], species, "x", sl=sl)**2, axis=avnp)
#     TeLy[i] = np.mean(o.getUth(time[i], species, "y", sl=sl)**2, axis=avnp)

#%%
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:],np.mean(Err,axis=0),color="k",label=r"$<(W_E-\mathcal{E}_{PIC})/\mathcal{E}_{PIC}>$")

sub1.set_xlim(min(t),max(t))
sub1.set_ylim(1e-3,10)

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_yscale("log")
sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:], np.mean(I_workE_centered_X,axis=0), color="r",label=r"$<W_{Ex}>$")
sub1.plot(t[1:], np.mean(I_workE_centered_Y,axis=0), color="g",label=r"$<W_{Ey}>$")
sub1.plot(t[1:], np.mean(I_workE_centered_Z,axis=0), color="b",label=r"$<W_{Ez}>$")

sub1.plot(t,np.mean(ene,axis=0),color="k",label=r"$<\mathcal{E}_{PIC}>$")

sub1.plot(t[1:],np.mean(I_workE_centered,axis=0),color="cyan",linestyle="--",label=r"$<\sum W_E>$")

# sub1.plot(time,TeLx,color="orange",label=r"$T_{ex}$")
# sub1.plot(time,TeLy,color="orange",linestyle="--",label=r"$T_{ey}$")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.legend(frameon=False)

"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

for i in range(len(ene)):

    sub1.plot(t[1:],I_workE_centered[i],color="r")

    sub1.plot(t[1:],ene[i,1:],linestyle="--",color="k")

sub1.set_xlim(min(t),max(t))
"""


