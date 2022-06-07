#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:59:28 2022

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
def cross(a,b,it):

    if it: return np.array([ a[1]*b[2]-a[2]*b[1],
                            -a[0]*b[2]+a[2]*b[0],
                             a[0]*b[1]-a[1]*b[0]]).T

    else: return np.array([ a[:,1]*b[:,2]-a[:,2]*b[:,1],
                           -a[:,0]*b[:,2]+a[:,2]*b[:,0],
                            a[:,0]*b[:,1]-a[:,1]*b[:,0]]).T

#----------------------------------------------
def boris_relativistic(p, ef, bf,
                       dt, charge, mass, it):
    #os-spec-push.f03 line 630 in osiris
    #p      = p^n-1/2
    #pnp    = p^n+1/2
    #pm     = p^-
    #pp     = p^+
    #pprime = p'

    fac = charge*dt/(2*mass)

    #get p_minus
    pm = p +  fac*ef

    #get p_prime
    tvec = fac / lorentz(pm)[...,None] * bf
    pprime = pm + cross(pm, tvec, it)

    #get p_plus
    svec = tvec * 2/(1+tvec[0]**2+tvec[1]**2+tvec[2]**2)
    pp = pm + cross(pprime, svec, it)

    #get next p
    pnp = pp + fac*ef

    return pnp, pp

#----------------------------------------------
def energy(p, mu):

    return (lorentz(p)-1) * mu

#----------------------------------------------
def lorentz(p):

    return np.sqrt(1+np.sum(p**2,axis=-1))

#----------------------------------------------
run ="CS3Dtrack"
# run="testTrackSingle"

spNorm = "eL"
o = osiris.Osiris(run,spNorm=spNorm)

species ="eL"

#----------------------------------------------
#index of macroparticle
sPart = 0
sTime = slice(None,None,1)
sl = (sPart,sTime)

#----------------------------------------------
massNorm = np.abs(o.rqm[o.sIndex(spNorm)])
mass = np.abs(o.rqm[o.sIndex(species)])
charge = np.sign(o.rqm[o.sIndex(species)])

#macroparticle, iteration
ene = o.getTrackData(species, "ene",sl=sl) * np.abs(mass)
t   = o.getTrackData(species, "t",  sl=sl)
p1  = o.getTrackData(species, "p1", sl=sl)
p2  = o.getTrackData(species, "p2", sl=sl)
p3  = o.getTrackData(species, "p3", sl=sl)
e1  = o.getTrackData(species, "E1", sl=sl)
e2  = o.getTrackData(species, "E2", sl=sl)
e3  = o.getTrackData(species, "E3", sl=sl)
b1  = o.getTrackData(species, "B1", sl=sl)
b2  = o.getTrackData(species, "B2", sl=sl)
b3  = o.getTrackData(species, "B3", sl=sl)

#----------------------------------------------
N = len(t)
dt = t[1]-t[0]

ef  = np.array([e1,e2,e3]).T
bf  = np.array([b1,b2,b3]).T
p   = np.array([p1,p2,p3]).T

energySim = energy(p, np.abs(mass))

#----------------------------------------------
#only integrate next time step
borisP = np.zeros((N,3))
borisP[0] = (p1[0],p2[0],p3[0])  #first value is initial condition (n-1/2 = -1/2)
#pp is momentum after first push and magnetic field rotation, fist value is n=0, centered momentum
borisP[1:], pp  = boris_relativistic(p[:-1], ef[:-1], bf[:-1],
                                     dt, charge, mass, it=False)
energyBoris = energy(borisP, np.abs(mass))

#----------------------------------------------
#integrate whole trajectory from initial conditions
borisPReIntegrated = np.zeros((N,3))
borisPReIntegrated[0] = (p1[0],p2[0],p3[0])
ppReIntegrated = np.zeros((N-1,3))
for i in range(0,N-1):
    borisPReIntegrated[i+1], ppReIntegrated[i] = boris_relativistic(borisPReIntegrated[i], ef[i], bf[i],
                                                                    dt, charge, mass, it=True)
energyBorisReIntegrated = energy(borisPReIntegrated, np.abs(mass))


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(energyBoris-energySim)/energySim)

sub1.set_yscale("log")

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(energyBorisReIntegrated-energySim)/energySim)

sub1.set_yscale("log")

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,borisPReIntegrated[:,0],color="r")
sub1.plot(t,borisPReIntegrated[:,1],color="g")
sub1.plot(t,borisPReIntegrated[:,2],color="b")

sub1.plot(t,p1,color="k",linestyle="--")
sub1.plot(t,p2,color="orange",linestyle="--")
sub1.plot(t,p3,color="cyan",linestyle="--")


#----------------------------------------------
#from sim and first push from data
workE_p = np.zeros((N,3))
workE_pp = np.zeros((N,3))

#divide by two since half time step each
workE_p[1:]  = p[:-1] /lorentz(p[:-1]) [...,None] *ef[:-1]*charge /2 #work first push
workE_pp[1:] = pp     /lorentz(pp)     [...,None] *ef[:-1]*charge /2 #work second push

totWorkE = np.sum(workE_p,axis=-1)+np.sum(workE_pp,axis=-1)
tot_int_WorkE = np.cumsum(totWorkE)*dt + energySim[0]

#----------------------------------------------
#reintegrated data and first push from reintegration
workE_borisPReIntegrated = np.zeros((N,3))
workE_ppReIntegrated = np.zeros((N,3))

workE_borisPReIntegrated[1:]  = borisPReIntegrated[:-1] /lorentz(borisPReIntegrated[:-1]) [...,None] *ef[:-1]*charge /2 #work first push
workE_ppReIntegrated[1:] = ppReIntegrated     /lorentz(ppReIntegrated)     [...,None] *ef[:-1]*charge /2 #work second push

totWorkEReIntegrated = np.sum(workE_borisPReIntegrated,axis=-1)+np.sum(workE_ppReIntegrated,axis=-1)
tot_int_WorkEReIntegrated = np.cumsum(totWorkEReIntegrated)*dt + energySim[0]

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.sum(workE_p,axis=-1),color="b")
sub1.plot(t,np.sum(workE_pp,axis=-1),color="g")

sub1.plot(t,totWorkE,color="r")

sub1.plot(t+dt/2,np.gradient(energySim,dt),color="k",linestyle="--")


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.sum(workE_borisPReIntegrated,axis=-1),color="b")
sub1.plot(t,np.sum(workE_ppReIntegrated,axis=-1),color="g")

sub1.plot(t,totWorkE,color="r")

sub1.plot(t+dt/2,np.gradient(energySim,dt),color="k",linestyle="--")



#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,tot_int_WorkE,color="r")
sub1.plot(t,tot_int_WorkEReIntegrated,color="b")
sub1.plot(t,energySim,color="k",linestyle="--")


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(tot_int_WorkE-energySim)/energySim)
sub1.set_yscale("log")
