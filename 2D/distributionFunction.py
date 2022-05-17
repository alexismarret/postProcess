#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 19:14:50 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import parallelFunctions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrRaw"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
st = slice(None,None,1)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sy]
time = o.getTimeAxis(species="iL", raw=True)[st]

#----------------------------------------------
fitFactor = 0.99
pStep = 1

show=True          #draw figures of fit
helpPos=False      #figure helper for region to consider

#filter criterion
window=False         #macroparticles within given position interval
density=False       #macroparticles within cells with density condition
current=True       #macroparticles within cells with current condition

#----------------------------------------------
if window or helpPos:
    #range of positions [c/wpi]
    #in underdense region
    posX = [0,3]
    posY = [3,3.5]
    #in overdense region
    posX = [0,3]
    posY = [3.5,4]

    #grid indexes corresponding to interval of positions wanted
    ipos = [np.where(x==posX[0])[0][0],
            np.where(x==posX[1])[0][0]]
    jpos = [np.where(x==posY[0])[0][0],
            np.where(x==posY[1])[0][0]]

#----------------------------------------------
#figure helper for region to consider
if helpPos:
    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    sub1.set_xlim(min(x),max(x))
    sub1.set_ylim(min(y),max(y))

    sub1.locator_params(nbins=20,axis='y')
    sub1.locator_params(nbins=20,axis='x')

    sub1.add_patch(Rectangle((posX[0],posY[0]),
                              posX[1]-posX[0],
                              posY[1]-posY[0],
                              fc='none',color="k"))

#----------------------------------------------
#setup figure
if show:
    plotPath = o.path+"/plots/distribUd"
    o.setup_dir(plotPath)
    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    sub1.set_xscale("log")
    sub1.set_yscale("log")

    sub1.set_xlim(0.3,0.6)
    sub1.set_ylim(1,1e5)

    sub1.set_xlabel(r"$v\ [c]$")

#----------------------------------------------
#gaussian funtion for fit
def gaussian(X, amp, index, drift):
    #index == m/kBT
    gauss = amp*np.exp(-0.5*index*(X-drift)**2)

    return gauss

T  = np.zeros(len(time))
TA = np.zeros(len(time))

ufl0 = o.ufl[o.sIndex("iL")][0]
vInit = ufl0 / np.sqrt(1+ufl0**2)

#----------------------------------------------
for i in range(len(time)):

    #----------------------------------------------
    #get macroparticles data, skip if none
    x1 = o.getRaw(time[i], "iL", "x1")
    try: len(x1)
    except TypeError: continue
    x2 = o.getRaw(time[i], "iL", "x2")

    #index of macroparticles cell
    gi, gj = o.findCell((x1,x2))

    p1 = o.getRaw(time[i], "iL", "p1")
    p2 = o.getRaw(time[i], "iL", "p2")
    p3 = o.getRaw(time[i], "iL", "p3")
    lorentz = np.sqrt(1+p1**2+p2**2+p3**2)

    p1/=lorentz

    #----------------------------------------------
    if window:
        #index of macroparticles in the wanted position interval
        indXp = np.where((gi>=ipos[0]) & (gi<=ipos[1]))[0]
        indYp = np.where((gj>=jpos[0]) & (gj<=jpos[1]))[0]

        cond = np.nonzero(np.in1d(indXp,indYp))[0]

    elif density:
        ni = o.getCharge(time[i], "iL")
        mask = np.ma.getmask(np.ma.masked_where(ni > o.n0[o.sIndex("iL")],
                                                ni, copy=False))
        cond = mask[gi,gj]

    elif current:
        J = o.getTotCurrent(time[i], "x")
        #true when condition met
        mask = np.ma.getmask(np.ma.masked_where(J > 0,
                                                J, copy=False))
        #keep macroparticles in cell where mask is true
        cond = mask[gi,gj]

    idx = np.nonzero(cond)[0][::pStep]

    #----------------------------------------------
    #generate histogram
    histogram, bins = np.histogram(p1[idx],bins=200)
    bins = bins[:-1]
    mf = 10000  #number of iterations max for fit

    #only fit thermal component
    limitV = fitFactor * np.mean(p1)
    filterThermal = (bins > limitV)
    #fit the data
    Famp, Findex, Fdrift  = curve_fit(gaussian,
                                      bins[filterThermal],
                                      histogram[filterThermal],
                                      p0=[np.max(histogram[filterThermal]),
                                          np.std(histogram[filterThermal]),
                                          vInit],
                                      maxfev=mf)[0]
    maxw = Famp*np.exp(-0.5*Findex*(bins-Fdrift)**2)

    #fit all the data
    FAamp, FAindex, FAdrift  = curve_fit(gaussian,
                                      bins,
                                      histogram,
                                      p0=[np.max(histogram),
                                          np.std(histogram),
                                          vInit],
                                      maxfev=mf)[0]
    Amaxw = FAamp*np.exp(-0.5*FAindex*(bins-FAdrift)**2)

    T[i] = 1/Findex
    TA[i] = 1/FAindex

    #----------------------------------------------
    if show:
        #plot
        l1 = sub1.axvline(limitV,color="gray",linestyle="--",linewidth=0.7)

        l2 = sub1.plot(bins,histogram,color="r")
        l4 = sub1.plot(bins,Amaxw,color="gray",linestyle="--")
        l3 = sub1.plot(bins,maxw,color="k")

        l5 = sub1.fill_between(bins,histogram,maxw,color="grey",alpha=0.5)

        if i==0: sub1.legend(loc='upper left')
        sub1.set_title(r"$t={t}\ [\Omega_0^{{-1}}]$".format(t=round(time[i],1)))

        plt.pause(1e-9)
        plt.savefig(plotPath+ "/plot-{i}-time-{t}.png".format(i=i,t=time[i]),dpi="figure")
        sub1.lines.remove(l1)
        sub1.lines.remove(l2[0])
        sub1.lines.remove(l3[0])
        sub1.lines.remove(l4[0])
        l5.remove()

if show: plt.close()

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
# sub1.plot(time,T,color="r")
# sub1.plot(time,TA,color="b")
sub1.set_ylim(1,1.4)
sub1.plot(time,TA/T,color="b")
