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

show=True       #draw figures of fit
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
def maxwellian(X, n, vth, vDrift):

    #vth == sqrt(kBT/m)
    gauss = n/(np.sqrt(2*np.pi)*vth) * np.exp(-0.5*((X-vDrift)/vth)**2)

    return gauss

#----------------------------------------------
def fitDistrib(ly, yStep, gj, p):

    nbins = 100
    mf = 10000

    rgeY = range(0,ly-yStep,yStep)
    vth = np.zeros(len(rgeY))

    for i,r in enumerate(rgeY):

        #macroparticles in row or group of row
        idx = np.nonzero((gj >= r) & (gj < r+yStep))[0]     #slowest operation


        #----------------------------------------------
        #generate histogram
        h, b = np.histogram(p[idx],bins=nbins)


        vth[i] = curve_fit(maxwellian, b[:-1], h, p0=[(b[1]- b[0])*np.max(h),
                                                       b[-1]-b[0],
                                                       vInit],
                                                        maxfev=mf)[0][0]

        # #----------------------------------------------
        # n, vth, vDrift = curve_fit(maxwellian, b[:-1], h, p0=[(b[1]- b[0])*np.max(h),
        #                                                        b[-1]-b[0],
        #                                                        vInit],
        #                                                   maxfev=mf)[0]
        # b = b[:-1]
        # fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)
        # sub1.plot(b,h,color="r")

        # maxw = n/(np.sqrt(2*np.pi)*vth) * np.exp(-0.5*((b-vDrift)/vth)**2)

        # sub1.plot(b,maxw,color="k")


        # db = b[-1]-b[0]
        # no = np.max(h*db)
        # vstd = db/10

        # print(no,vstd,vDrift)
        # print(n, vth, vDrift)

        # test = no/(np.sqrt(2*np.pi)*vstd) * np.exp(-0.5*((b-vInit)/vstd)**2)
        # sub1.plot(b,test,color="b")

    return vth

#----------------------------------------------
#https://www.geeksforgeeks.org/find-first-and-last-positions-of-an-element-in-a-sorted-array/
# if x is present in arr[] then
# returns the index of LAST occurrence
# of x in arr[0..n-1]
def last(arr, low, high, x, n) :
    if (high >= low) :
        mid = low + (high - low) // 2
        if (( mid == n - 1 or x < arr[mid + 1]) and arr[mid] == x) :
            return mid
        elif (x < arr[mid]) :
            return last(arr, low, (mid - 1), x, n)
        else :
            return last(arr, (mid + 1), high, x, n)

    return


#----------------------------------------------
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
    try: N = len(x1)
    except TypeError: continue
    x2 = o.getRaw(time[i], species, "x2")

    #sort macroparticles along x position
    argsort = np.argsort(x1)

    p1 = o.getRaw(time[i], species, "p1")
    p2 = o.getRaw(time[i], species, "p2")
    p3 = o.getRaw(time[i], species, "p3")
    lorentz = np.sqrt(1+p1**2+p2**2+p3**2)

    p1/=lorentz
    p1 = p1[argsort]

    #index of macroparticles cell, sorted along x
    gi, gj = o.findCell((x1[argsort],x2[argsort]))

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
        except: cond = np.ones(p1.shape,dtype=bool)
        # Ti  = np.ma.mean(np.ma.masked_where(~mask, Temp, copy=False))
        # TiN = np.ma.mean(np.ma.masked_where( mask, Temp, copy=False))

    elif current:
        J = o.getTotCurrent(time[i], "x")
        #true when condition met
        mask = np.ma.getmask(np.ma.masked_where(J > 0, J, copy=False))
        #keep macroparticles in cell where mask is true
        try: cond = mask[gi,gj]
        except: cond = np.ones(p1.shape,dtype=bool)
        #masked when mask is true, so invert to get values when mask is true
        # Ti  = np.ma.mean(np.ma.masked_where(~mask, Temp, copy=False))  #temp in filament
        # TiN = np.ma.mean(np.ma.masked_where( mask, Temp, copy=False))  #temp out of filament

    #indexes of macroparticles fulfilling the condition
    # idx = np.nonzero(cond)[0][::pStep]
    # idx_not = np.nonzero(~cond)[0][::pStep]



    #----------------------------------------------
    #would be better to get fit in each cell separetely, and divide by temp diagnostic

    lx = 512
    ly = 512

    #average over several cells
    #if not a dividor, remaining cells are ignored
    xStep = 2
    yStep = 2

    #find indexes of last particles to be in a given row
    f = [last(gi, 0, N-1, x, N) for x in range(o.grid[0])]

    #slices of all particles in a given row, or number of row given by xStep
    #sl[0] : gives all indexes of macroparticles in row 0 to xStep-1 included
    rgeX = range(xStep-1,o.grid[0]-xStep,xStep)

    sl = [None]*(len(rgeX)+1)
    sl[0] = slice(0,f[xStep-1]+1)

    for i,r in enumerate(rgeX):
        sl[i+1] = slice(f[r]+1,f[r+xStep]+1)


    it = ((ly, yStep, gj[s], p1[s]) for s in sl)

    import time as ti
    start = ti.time()
    vth = pf.parallel(fitDistrib, it, o.nbrCores, noInteract=False)
    print(ti.time()-start)

    # import time as ti
    # start = ti.time()
    # vth = fitDistrib(ly, yStep, gj[sl[0]], p1[sl[0]])
    # print(ti.time()-start)


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
"""
