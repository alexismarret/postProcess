#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:37:38 2022

@author: alexis
"""


#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------
run = ['CS3DMu64Ma20theta10','CS3DMu64Ma40theta10',
       'CS3DMu64Ma20theta80','CS3DMu64Ma40theta80',
       'CS3DMu64Ma200theta10',
       'CS3DMu64noB','CS3Dtrack']

o = osiris.Osiris(run[0],spNorm="iL")

ind = 50
sx = slice(0,ind,1)
sy = slice(0,ind,1)
sz = slice(0,ind,1)
sl = (sx,sy,sz)

st = slice(0,-1,1) #ignore last time step of m=64 runs, some were not saved in time
time = o.getTimeAxis()[st]

avnp = (0,1,2)     #averaging axis
dtype = 'float32'  #precision for mean calculation

#----------------------------------------------
ratioX = np.zeros((len(run),len(time)))
Te     = np.zeros((len(run),len(time)))
Ti     = np.zeros((len(run),len(time)))
Jtot   = np.zeros((len(run),len(time)))

for j in range(len(run)):

    print(run[j])

    #----------------------------------------------
    if run[j]=="CS3Dtrack":
        o = osiris.Osiris(run[j],spNorm="iL",globReduced=False)
        mass = o.rqm[o.sIndex("iL")]
        eps = o.uth[o.sIndex("iL"),0]**2*mass

        time = o.getTimeAxis()
        ratioX_mu32 = np.zeros(len(time))
        Te_mu32     = np.zeros(len(time))
        Ti_mu32     = np.zeros(len(time))
        Jtot_mu32   = np.zeros(len(time))
        for i in range(len(time)):

            # print(time[i])
            Jtot_mu32[i] = np.mean(o.getTotCurrent(time[i],"y",sl=sl)**2, axis=avnp, dtype=dtype)
            Te_ = o.getUth(time[i], "eL", "x", sl=sl)**2
            Ti_ = o.getUth(time[i], "iL", "x", sl=sl)**2 * mass
            # Te_ = np.ma.masked_where(Te_==0, Te_, copy=False)
            # Ti_ = np.ma.masked_where(Ti_==0, Ti_, copy=False)

            Te_mu32[i] = np.mean(Te_, axis=avnp, dtype=dtype)
            Ti_mu32[i] = np.mean(Ti_, axis=avnp, dtype=dtype)

            ratioX_mu32[i] = np.mean(Te_/(Ti_+eps), axis=avnp, dtype=dtype)

    #----------------------------------------------
    else:
        o = osiris.Osiris(run[j],spNorm="iL",globReduced=True)
        mass = o.rqm[o.sIndex("iL")]
        eps = o.uth[o.sIndex("iL"),0]**2*mass

        time = o.getTimeAxis()[st]
        for i in range(len(time)):

            # print(time[i])
            Jtot[j,i] = np.mean(o.getTotCurrent(time[i],"y",sl=sl)**2, axis=avnp, dtype=dtype)
            Te_ = o.getUth(time[i], "eL", "x", sl=sl)**2
            Ti_ = o.getUth(time[i], "iL", "x", sl=sl)**2 * mass
            # Te_ = np.ma.masked_where(Te_==0, Te_, copy=False)
            # Ti_ = np.ma.masked_where(Ti_==0, Ti_, copy=False)

            Te[j,i] = np.mean(Te_, axis=avnp, dtype=dtype)
            Ti[j,i] = np.mean(Ti_+eps, axis=avnp, dtype=dtype)

            ratioX[j,i] = np.mean(Te_/(Ti_+eps), axis=avnp, dtype=dtype)



#%%
plt.close("all")
#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 7, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

#----------------------------------------------
colors=["r","g","b","c","m","y"]
linestyles=["solid","dotted","dashed","dashdot",(0,(1,10)),(0,(5,10))]

for j in range(len(run)-1):
    o = osiris.Osiris(run[j],spNorm="iL",globReduced=True)
    time = o.getTimeAxis()[st]

    indKinkSat = np.where(Jtot[j]==np.max(Jtot[j]))[0][0]
    sub1.axvline(time[indKinkSat],color=colors[j],linewidth=0.7,linestyle=linestyles[j])

    sub1.plot(time,ratioX[j],label=run[j],color=colors[j])

#----------------------------------------------
o = osiris.Osiris("CS3Dtrack",spNorm="iL",globReduced=False)
time = o.getTimeAxis()

indKinkSat = np.where(Jtot_mu32==np.max(Jtot_mu32))[0][0]
sub1.axvline(time[indKinkSat],color="k",linewidth=0.7)

sub1.plot(time,ratioX_mu32,color="k",label="CS3Dmu32noB")


sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$T_e/T_i$")
sub1.legend(frameon=False)
sub1.set_yscale("log")




#----------------------------------------------
fig, (sub1,sub2) = plt.subplots(2,figsize=(4.1,2.8),dpi=300)

o = osiris.Osiris(run[0],spNorm="iL",globReduced=True)
time = o.getTimeAxis()[st]
sub1.axvline(time[1],color="gray",linestyle="--",linewidth=0.7)
sub2.axvline(time[1],color="gray",linestyle="--",linewidth=0.7)

#----------------------------------------------
colors=["r","g","b","c","m","y"]
for j in range(len(run)-1):
    o = osiris.Osiris(run[j],spNorm="iL",globReduced=True)
    time = o.getTimeAxis()[st]

    indKinkSat = np.where(Jtot[j]==np.max(Jtot[j]))[0][0]
    sub1.axvline(time[indKinkSat],color=colors[j],linewidth=0.7,linestyle=linestyles[j])
    sub2.axvline(time[indKinkSat],color=colors[j],linewidth=0.7,linestyle=linestyles[j])

    print(run[j],indKinkSat)
    sub1.plot(time, Te[j],color=colors[j],linestyle="--",label=run[j])
    sub2.plot(time, Ti[j],color=colors[j])


#----------------------------------------------
o = osiris.Osiris("CS3Dtrack",spNorm="iL",globReduced=False)
time = o.getTimeAxis()

indKinkSat = np.where(Jtot_mu32==np.max(Jtot_mu32))[0][0]
sub1.axvline(time[indKinkSat],color="k",linewidth=0.7)
sub2.axvline(time[indKinkSat],color="k",linewidth=0.7)

sub1.plot(time, Te_mu32,color="k",linestyle="--",label="CS3Dmu32noB")
sub2.plot(time, Ti_mu32,color="k")

sub2.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$T_e\ [m_ec^2]$")
sub2.set_ylabel(r"$T_i\ [m_ec^2]$")
sub1.legend(frameon=False)
