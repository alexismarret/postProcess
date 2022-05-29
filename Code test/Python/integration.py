#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:23:50 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import trackParticles as tr

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 0.9,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
# run  ="CS3Dtrack"
# run ="CS3Dtrack"
run="testTrackSingle"
spNorm = "eL"
o = osiris.Osiris(run,spNorm=spNorm)

species ="eL"

#----------------------------------------------
#index of macroparticle
sPart = slice(None,None,1000)
sTime = slice(None,None,1)
sl = (sPart,sTime)

#----------------------------------------------
muNorm  = o.rqm[o.sIndex(spNorm)]
mu      = o.rqm[o.sIndex(species)]
pCharge = mu / np.abs(mu)

#macroparticle, iteration
ene = o.getTrackData(species, "ene",sl=sl)[:,:-1] * np.abs(mu)
t   = o.getTrackData(species, "t",  sl=sl)[:,:-1]
p1  = o.getTrackData(species, "p1", sl=sl)[:,1:]
p2  = o.getTrackData(species, "p2", sl=sl)[:,1:]
p3  = o.getTrackData(species, "p3", sl=sl)[:,1:]
e1  = o.getTrackData(species, "E1", sl=sl)[:,:-1]
e2  = o.getTrackData(species, "E2", sl=sl)[:,:-1]
e3  = o.getTrackData(species, "E3", sl=sl)[:,:-1]

x = o.getAxis("x")
y = o.getAxis("y")
z = o.getAxis("z")

dt = (t[0,1]-t[0,0])

imax = np.where(ene[:,-1]==np.min(ene[:,-1]))[0][0]  #index of least energetic particle

#get velocity
lorentz = np.sqrt(1+p1**2+p2**2+p3**2)
p1 /= lorentz
p2 /= lorentz
p3 /= lorentz

#----------------------------------------------
work1 = pCharge * e1*p1
work2 = pCharge * e2*p2
work3 = pCharge * e3*p3
work = work1+work2+work3

intWork1 = cumulative_trapezoid(work1,t,axis=-1,initial=0)
intWork2 = cumulative_trapezoid(work2,t,axis=-1,initial=0)
intWork3 = cumulative_trapezoid(work3,t,axis=-1,initial=0)

intWork = intWork1 + intWork2 + intWork3

"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)


# sub1.plot(t[imax],ene[imax],color="r")
# sub1.plot(t[imax], intWork1[imax] ,color="b")
# sub1.plot(t[imax], intWork2[imax] ,color="orange")
# sub1.plot(t[imax], intWork3[imax] ,color="g")
# sub1.plot(t[imax], intWork[imax] + ene[imax,0],color="k",linestyle="dotted")

# sub1.plot(t[imax,:-2],a1[imax],color="r",linestyle="dotted")
# sub1.plot(t[imax,:-2],a2[imax],color="r",linestyle="dotted")
# sub1.plot(t[imax,:-2],a3[imax],color="r",linestyle="dotted")
# sub1.plot(t[imax,:-2],a[imax]+ ene[imax,0],color="r",linestyle="dotted")

diff1 = (intWork[imax]+ene[imax,0] - ene[imax]) / ene[imax,0]
sub1.plot( diff1, color="orange",linestyle="-")

"""

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)


d_ene_dt = (ene[:,1:] - ene[:,:-1])/dt
d_ene_dt_np = np.gradient(ene,dt,axis=-1)

sub1.plot(t[imax,:-1],d_ene_dt[imax],color="k",label=r"$d_t\ ene$",marker="x")

#shifted by dt/2 because of derivative, value is in between two data points
# sub1.plot(t[imax]-dt/2,d_ene_dt_np[imax],color="orange",label=r"$d_t\ ene$",marker="x")

# sub1.plot(t[imax],ene[imax]-ene[imax,0],color="b",label=r"$ene$",marker="x")

sub1.plot(t[imax],work[imax],color="orange",label=r"$W$")
sub1.legend(frameon=False)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[imax],ene[imax]-ene[imax,0],color="k")


# def integral(data,delta):

#     return delta*np.cumsum(data[:,:-1],axis=-1)

# int_dt_ene = integral(d_ene_dt_np,dt)
# int_work   = integral(work,dt)

# sub1.plot(t[imax,1:],int_dt_ene[imax],color="k")
# sub1.plot(t[imax,1:],int_work[imax],color="orange")
sub1.plot(t[imax],intWork[imax],color="orange")











