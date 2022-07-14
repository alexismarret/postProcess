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
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

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

for j in range(len(run)):

    print(run[j])

    #----------------------------------------------
    if run[j]=="CS3Dtrack":
        o = osiris.Osiris(run[j],spNorm="iL",globReduced=False)

        mass = o.rqm[o.sIndex("iL")]
        eps  = o.uth[o.sIndex("iL"),0]  #avoid div/0

        time = o.getTimeAxis()
        ratioX_mu32 = np.zeros(len(time))
        for i in range(len(time)):

            # print(time[i])
            Te_Ti = (o.getUth(time[i], "eL", "x", sl=sl) /
                    (o.getUth(time[i], "iL", "x", sl=sl)+eps))

            ratioX_mu32[i] = np.mean((Te_Ti)**2, axis=avnp, dtype=dtype) / mass

    #----------------------------------------------
    else:
        o = osiris.Osiris(run[j],spNorm="iL",globReduced=True)

        mass = o.rqm[o.sIndex("iL")]
        eps  = o.uth[o.sIndex("iL"),0]  #avoid div/0

        time = o.getTimeAxis()[st]
        for i in range(len(time)):

            # print(time[i])
            Te_Ti = (o.getUth(time[i], "eL", "x", sl=sl) /
                    (o.getUth(time[i], "iL", "x", sl=sl)+eps))

            ratioX[j,i] = np.mean((Te_Ti)**2, axis=avnp, dtype=dtype) / mass



#%%
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

#----------------------------------------------
for j in range(len(run)-1):
    o = osiris.Osiris(run[j],spNorm="iL",globReduced=True)
    time = o.getTimeAxis()[st]

    sub1.plot(time,ratioX[j],label=run[j])

#----------------------------------------------
o = osiris.Osiris("CS3Dtrack",spNorm="iL",globReduced=False)
time = o.getTimeAxis()
sub1.plot(time,ratioX_mu32,label="CS3Dmu32noB")


sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$T_e/T_i$")
sub1.legend(frameon=False)
sub1.set_yscale("log")
