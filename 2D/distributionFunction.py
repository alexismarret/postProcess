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
from scipy.stats import skew

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import parallelFunctions as pf
import time as ti

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")
plt.switch_backend('Qt5Agg')

#----------------------------------------------
#https://www.geeksforgeeks.org/find-first-and-last-positions-of-an-element-in-a-sorted-array/
# if x is present in arr[] then
# returns the index of LAST occurrence
# of x in arr[0..n-1]
def last(arr, low, high, x, n):
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
#gaussian funtion for fit
def maxwellian(X, n, vth, vDrift):

    #vth == sqrt(kBT/m)
    # gauss = n/(np.sqrt(2*np.pi)*vth) * np.exp(-0.5*((X-vDrift)/vth)**2)

    #n is amplitude, mix of vth and density, but not important here so free parameter
    #only vDrfit and vth are important, simplify fitting and avoid degeneracy
    #vth == sqrt(kBT/m)
    #factor exp(1/2) goes into n
    #can return vth negative

    return n * np.exp(-((X-vDrift)/vth)**2)


#----------------------------------------------
def sliceGrid(g, l, step):

    N = len(g)

    #find indexes of last particles to be in a given row, from sorted array g
    f = [last(g, 0, N-1, x, N) for x in range(g[-1]+1)]

    #slices of all particles in a given row or column, or number of row or column given by xStep
    #None if no macroparticles
    #sl[0] : gives all indexes of macroparticles in row or column 0 to xStep-1 included
    sl = [None]*(l//step)

    try: sl[0] = slice(0,f[step-1]+1)
    except TypeError: sl[0] = None  #in case no particles in given row or column

    rgeX = range(step-1,l-step,step)
    for j,r in enumerate(rgeX):
        try: sl[j+1] = slice(f[r]+1,f[r+step]+1)
        except TypeError: continue
        except IndexError: continue

    return sl

#----------------------------------------------
def fitDistrib(ly, yStep, gj, p, time=None, j=None, sub=None, check=False):

    mf = 1000  #max number of iterations for fit
    minN = 0   #min number of macroparticles required to attempt postprocessing
    nbins = 70  #number of bins of histogram
    showGuess = False
    showLim =False
    fit = False
    fitFactor = 0.05

    argsort = np.argsort(gj)
    p = p[argsort]
    sl = sliceGrid(gj[argsort], ly, yStep)

    vth = np.empty(ly//yStep)
    vth.fill(np.nan)
    skewness = np.empty(ly//yStep)
    skewness.fill(np.nan)

    for k,s in enumerate(sl):

        if (len(p[s])<minN):
            # print("not enough parts")
            continue

        #----------------------------------------------
        h, b = np.histogram(p[s],bins=nbins)
        skewness[k] = skew(h)

        if not fit: continue

        ma = np.max(h)
        idMax = np.where(h==ma)[0][0]
        p0 = [ma, np.std(b), b[idMax]]

        b = b[:-1]
        condMax = (b<b[idMax]*(1+fitFactor))
        condMin = (b>b[idMax]*(1-fitFactor))
        cond = condMax & condMin

        #----------------------------------------------
        if not check:
            try:
                vth[k] = curve_fit(maxwellian, b[cond], h[cond], p0=p0, maxfev=mf)[0][1]
            except RuntimeError:
                # print("No fit found")
                continue
            except TypeError:
                # print("not enoug bins")
                continue

        #----------------------------------------------
        elif check:

            #handle single or multiple subplots
            try: kjSub = sub[k,j]
            except TypeError: kjSub = sub

            #show initial guess
            if showGuess:
                test = maxwellian(b, p0[0], p0[1], p0[2])
                kjSub.plot(b,test,color="b",label=r"$Guess$")

            #display restricted window
            if showLim:
                diffMin = np.abs(b-b[idMax]*(1-fitFactor))
                diffMax = np.abs(b-b[idMax]*(1+fitFactor))
                idCondMin = np.where(diffMin==np.min(diffMin))[0][0]
                idCondMax = np.where(diffMax==np.min(diffMax))[0][0]

                # kjSub.axvline(b[idMax],color="gray",linestyle="--",linewidth=0.7)
                kjSub.axvline(b[idCondMin],color="gray",linestyle="--",linewidth=0.7)
                kjSub.axvline(b[idCondMax],color="gray",linestyle="--",linewidth=0.7)

            #display data
            kjSub.plot(b,h,color="orange",label=r"$Distribution$")

            #do fit
            try:
                n, vth[k], vDrift = curve_fit(maxwellian, b[cond], h[cond], p0=p0, maxfev=mf)[0]
            except RuntimeError:
                # print("No fit found")
                continue
            except TypeError:
                # print("not enoug bins")
                continue

            #plot fit
            maxw = maxwellian(b, n, vth[k], vDrift)
            kjSub.plot(b,maxw,color="k",linestyle="--",linewidth=1.5,label=r"$Fit$")

            # print(p0)
            # print(n, vth[k], vDrift)

            kjSub.text(1, 1.04,
                            r"$(Gx,Gy)=(%.0f,%.0f)$"
                            %(j,k),
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            transform=kjSub.transAxes)

            if (k,j) == (0,0):
                kjSub.text(0.35, 1.03,
                                r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time,
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                transform=kjSub.transAxes)
                kjSub.legend(frameon=False)

    #vth can be negative due to the fitting method, but absolute value is the same
    return np.abs(vth), skewness

#----------------------------------------------
#----------------------------------------------
# run  ="CS2DrmhrTrack"
run = "CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

species = "iL"

st = slice(None,None,1)
x    = o.getAxis("x")
y    = o.getAxis("y")
time = o.getTimeAxis(species=species, raw=True)[st]

#----------------------------------------------
stdf = 1      #filter standard deviation

helpPos=False      #figure helper for region to consider
do_filter=False    #filter histogram

#filter criterion
window=False      #macroparticles within given position interval
density=False       #macroparticles within cells with density condition
current=False       #macroparticles within cells with current condition

#fitting
parallel=True
check=False

lx = 512
ly = 512

#number of cells to consider for computing the global distribution function
#if not a dividor, remaining cells are ignored
xStep = 1
yStep = 1

Nx = lx//xStep
Ny = ly//yStep

slx = slice(None,lx,xStep)
sly = slice(None,ly,yStep)

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
ufl0 = o.ufl[o.sIndex(species)]
vInit = ufl0[0] / np.sqrt(1+ufl0[0]**2+ufl0[1]**2+ufl0[2]**2)

vth = np.zeros((len(time),lx//xStep,ly//yStep))
skewness = np.zeros((len(time),lx//xStep,ly//yStep))

#----------------------------------------------
for i in range(len(time)):
    print("time:",time[i])
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

    #----------------------------------------------
    if check:
        fig, sub = plt.subplots(Nx,Ny,figsize=(4.1,4.1),dpi=300)

        for Rsub in range(Ny):
            for Csub in range(Nx):
                #draw labels
                if Ny==1:
                    if Csub==0:    sub.set_ylabel(r"$count$")
                    if Rsub==Ny-1: sub.set_xlabel(r"$v$")
                else:
                    plotIndex = (Rsub,Csub)
                    if Csub==0:    sub[plotIndex].set_ylabel(r"$count$")
                    if Rsub==Ny-1: sub[plotIndex].set_xlabel(r"$v$")
    else: sub = None

    #----------------------------------------------
    sl = sliceGrid(gi, lx, xStep)
    it = ((ly, yStep, gj[s], p1[s]) for j,s in enumerate(sl))

    if parallel:
        start = ti.time()
        vth[i], skewness[i] = pf.parallel(fitDistrib, it, o.nbrCores, noInteract=True)
        print(ti.time()-start)

    else:
        for j,s in enumerate(sl):
            # start = ti.time()
            vth[i,j], skewness[i,j] = fitDistrib(ly, yStep, gj[s], p1[s], time[i], j, sub, check=check)
            # print(ti.time()-start)
            # print("")



#----------------------------------------------
#dump data to disk

if not check:
    o.writeHDF5(skewness, "skew")
    # o.writeHDF5(vth, "vth")
