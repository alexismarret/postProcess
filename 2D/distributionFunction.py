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
from scipy import signal

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
plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

species = "iL"

st = slice(None,None,1)
x    = o.getAxis("x")
y    = o.getAxis("y")
time = o.getTimeAxis(species=species, raw=True)[st]

#----------------------------------------------
pStep = 1

fitFactor = 1
mf = 100000  #number of iterations max for fit
stdf = 1      #filter standard deviation

show=False       #draw figures of fit
helpPos=False      #figure helper for region to consider
do_filter=False    #filter histogram

#filter criterion
window=False      #macroparticles within given position interval
density=True       #macroparticles within cells with density condition
current=False       #macroparticles within cells with current condition

#----------------------------------------------
if window or helpPos:
    #range of positions (units of x and y)
    posX = [0,10]
    posY = [0,10]

    #grid indexes corresponding to interval of positions wanted
    ipos = [np.where(x==posX[0])[0][0],
            np.where(x==posX[1])[0][0]]
    jpos = [np.where(x==posY[0])[0][0],
            np.where(x==posY[1])[0][0]]

#----------------------------------------------
#figure helper for region to consider
if helpPos:
    fig, sub1 = plt.subplots(1,figsize=(3,3),dpi=300)

    sub1.set_xlim(min(x),max(x))
    sub1.set_ylim(min(y),max(y))

    sub1.locator_params(nbins=20,axis='y')
    sub1.locator_params(nbins=20,axis='x')

    sub1.add_patch(Rectangle((posX[0],posY[0]),
                              posX[1]-posX[0],
                              posY[1]-posY[0],
                              fc='none',color="k"))

    raise ValueError

#----------------------------------------------
#setup figure
if show:
    plotPath = o.path+"/plots/distribIN"

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
TN  = np.zeros(len(time))
TAN = np.zeros(len(time))

ufl0 = o.ufl[o.sIndex(species)][0]
vInit = ufl0 / np.sqrt(1+ufl0**2)

#----------------------------------------------
for i in range(len(time)):

    #----------------------------------------------
    #get macroparticles data, skip if none
    x1 = o.getRaw(time[i], species, "x1")
    try: len(x1)
    except TypeError: continue
    x2 = o.getRaw(time[i], species, "x2")

    #index of macroparticles cell
    gi, gj = o.findCell((x1,x2))

    p1 = o.getRaw(time[i], species, "p1")
    p2 = o.getRaw(time[i], species, "p2")
    p3 = o.getRaw(time[i], species, "p3")
    lorentz = np.sqrt(1+p1**2+p2**2+p3**2)

    p1/=lorentz

    #get temperature from whole macroparticles distribution
    # Temp = o.getUth(time[i], species, "x")**2 *o.rqm[o.sIndex(species)]
    #----------------------------------------------
    if window:
        #index of macroparticles in the wanted position interval
        indXp = np.where((gi>=ipos[0]) & (gi<=ipos[1]))[0]
        indYp = np.where((gj>=jpos[0]) & (gj<=jpos[1]))[0]

        cond = np.nonzero(np.in1d(indXp,indYp))[0]
        print("Not implemented Temp in cell")
        raise ValueError

    elif density:
        ni = o.getCharge(time[i], species)
        mask = np.ma.getmask(np.ma.masked_where(ni > o.n0[o.sIndex(species)],
                                                ni, copy=False))
        try: cond = mask[gi,gj]
        except: cond = np.ones(x1.shape,dtype=bool)
        # Ti  = np.ma.mean(np.ma.masked_where(~mask, Temp, copy=False))
        # TiN = np.ma.mean(np.ma.masked_where( mask, Temp, copy=False))

    elif current:
        J = o.getTotCurrent(time[i], "x")
        #true when condition met
        mask = np.ma.getmask(np.ma.masked_where(J > 0, J, copy=False))
        #keep macroparticles in cell where mask is true
        try: cond = mask[gi,gj]
        except: cond = np.ones(x1.shape,dtype=bool)
        #masked when mask is true, so invert to get values when mask is true
        # Ti  = np.ma.mean(np.ma.masked_where(~mask, Temp, copy=False))  #temp in filament
        # TiN = np.ma.mean(np.ma.masked_where( mask, Temp, copy=False))  #temp out of filament

    #indexes of macroparticles fulfilling the condition
    idx = np.nonzero(cond)[0][::pStep]
    idx_not = np.nonzero(~cond)[0][::pStep]

    """
    #would be better to get fit in each cell separetely, and divide by temp diagnostic

    @numba.njit()
    def dostuff(lx,ly,mask,gi,gj,pStep):
        # for k in range(o.grid[0]):
        #     for l in range(o.grid[1]):
        for k in range(lx):
            for l in range(ly):
                print(k,l)
                if mask[k,l]:
                    idx = np.nonzero((gi==k) & (gj==l))[0][::pStep]
                else:
                    idx_not = np.nonzero((gi==k) & (gj==l))[0][::pStep]

    dostuff(o.grid[0],o.grid[1], mask, gi, gj, pStep)

    raise ValueError
    """

    #----------------------------------------------
    #generate histogram
    histogram, bins = np.histogram(p1[idx],bins=200)
    histogram_not, bins_not = np.histogram(p1[idx_not],bins=200)

    if do_filter:
        win=signal.gaussian(len(histogram),stdf)   #window
        winN=signal.gaussian(len(histogram_not),stdf)

        histogram = signal.convolve(histogram, win, mode='same') / sum(win)
        histogram_not = signal.convolve(histogram_not, winN, mode='same') / sum(win)

    bins = bins[:-1]
    bins_not = bins_not[:-1]

    #----------------------------------------------
    #fit with macroparticles fulfilling the condition
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

    #----------------------------------------------
    #fit with the others
    filterThermal_not = (bins_not > limitV)
    #fit the data
    FampN, FindexN, FdriftN  = curve_fit(gaussian,
                                         bins_not[filterThermal_not],
                                         histogram_not[filterThermal_not],
                                         p0=[np.max(histogram_not[filterThermal_not]),
                                             np.std(histogram_not[filterThermal_not]),
                                             vInit],
                                         maxfev=mf)[0]

    maxwN = FampN*np.exp(-0.5*FindexN*(bins_not-FdriftN)**2)

    #fit all the data
    FAampN, FAindexN, FAdriftN  = curve_fit(gaussian,
                                            bins_not,
                                            histogram_not,
                                            p0=[np.max(histogram_not),
                                                np.std(histogram_not),
                                                vInit],
                                            maxfev=mf)[0]

    AmaxwN = FAampN*np.exp(-0.5*FAindexN*(bins_not-FAdriftN)**2)

    #----------------------------------------------
    T[i]   = o.rqm[o.sIndex(species)]/Findex    #in filaments, only maxwellian part
    # TA[i]  = Ti                                 #in filaments, from whole distribution
    TA[i] = o.rqm[o.sIndex(species)]/FAindex
    TN[i]  = o.rqm[o.sIndex(species)]/FindexN   #outside filaments, only maxwellian part
    # TAN[i] = TiN                                #outside filaments, from whole distribution
    TAN[i] = o.rqm[o.sIndex(species)]/FAindexN

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

sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)
# sub1.set_ylim(1,1.8)
sub1.plot(time,TA/T,color="b")
sub1.plot(time,TAN/TN,color="r")

ratio = TA/T / (TAN/TN)
sub1.plot(time,ratio,color="k")
