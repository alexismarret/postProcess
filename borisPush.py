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

    fac = charge/mass * dt/2.

    #get p_minus
    pm = p + fac*ef

    #get p_prime
    tvec = fac / lorentz(pm)[...,None] * bf
    pprime = pm + cross(pm, tvec, it)

    #get p_plus
    svec = tvec * 2./(1+tvec[0]**2+tvec[1]**2+tvec[2]**2)
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

species ="iL"

#----------------------------------------------
#index of macroparticle
sPart = 0
sTime = slice(None,None,1)
sl = (sPart,sTime)

#----------------------------------------------
mass   = np.abs (o.rqm[o.sIndex(species)])
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
#only integrate next time step then come back to simulation data to integrate the next one
p_boris = np.zeros((N,3))
p_boris[0] = (p1[0],p2[0],p3[0])  #first value is initial condition (n-1/2 = -1/2)

#pp is momentum after first push and magnetic field rotation, fist value is n=0, centered momentum
p_boris[1:], pp  = boris_relativistic(p[:-1], ef[:-1], bf[:-1],
                                      dt, charge, mass, it=False)
energyBoris = energy(p_boris, mass)   #energy after the two half pushes

#----------------------------------------------
#integrate whole trajectory from initial conditions, reinject previous solution
p_boris_ri = np.zeros((N,3))
p_boris_ri[0] = (p1[0],p2[0],p3[0])

pp_ri = np.zeros((N-1,3))
for i in range(0,N-1):
    p_boris_ri[i+1], pp_ri[i] = boris_relativistic(p_boris_ri[i], ef[i], bf[i],
                                                   dt, charge, mass, it=True)
energyBorisReIntegrated = energy(p_boris_ri, mass)

#----------------------------------------------
#work from sim trajectory
workE_p = np.zeros((N,3))
workE_pp = np.zeros((N,3))

#divide by two since half time step each
workE_p[1:]  = p[:-1] /lorentz(p[:-1]) [...,None] *ef[:-1]*charge /2 #work first push
workE_pp[1:] = pp     /lorentz(pp)     [...,None] *ef[:-1]*charge /2 #work second push

totWorkE_sim = np.sum(workE_p,axis=-1) + np.sum(workE_pp,axis=-1) #sum over two half pushes and over components
I_totWorkE_sim = np.cumsum(totWorkE_sim)*dt + energySim[0]

#----------------------------------------------
#work from reintegrated trajectory
workE_p_boris_ri = np.zeros((N,3))
workE_pp_ri = np.zeros((N,3))

workE_p_boris_ri[1:]  = p_boris_ri[:-1] /lorentz(p_boris_ri[:-1]) [...,None] *ef[:-1]*charge /2 #work first push
workE_pp_ri[1:] = pp_ri/lorentz(pp_ri)[...,None] *ef[:-1]*charge /2 #work second push

totWork_p_boris_ri = np.sum(workE_p_boris_ri,axis=-1)+np.sum(workE_pp_ri,axis=-1)
I_totWork_p_boris_ri = np.cumsum(totWork_p_boris_ri)*dt + energySim[0]


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(energyBoris-energySim)/energySim,
          label=r"$(\mathbf{E}_{boris\ next\ only}-\mathbf{E}_{sim})/\mathbf{E}_{sim}$")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(energyBorisReIntegrated-energySim)/energySim,
          label=r"$(\mathbf{E}_{boris\ reintegrated}-\mathbf{E}_{sim})/\mathbf{E}_{sim}$")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,p_boris_ri[:,0],color="r",label=r"$p1_{boris\ ri}$")
sub1.plot(t,p_boris_ri[:,1],color="g",label=r"$p1_{boris\ ri}$")
sub1.plot(t,p_boris_ri[:,2],color="b",label=r"$p1_{boris\ ri}$")

sub1.plot(t,p1,color="k",linestyle="--",label=r"$p1_{boris\ sim}$")
sub1.plot(t,p2,color="orange",linestyle="--",label=r"$p1_{boris\ sim}$")
sub1.plot(t,p3,color="cyan",linestyle="--",label=r"$p1_{boris\ sim}$")

plt.xlabel(r'$t$')
sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.sum(workE_p,axis=-1),color="cyan",label=r"$\vec v_{sim}\cdot\vec E$")
sub1.plot(t,np.sum(workE_pp,axis=-1),color="g",linestyle="--",label=r"$\vec v_{sim\ centered}\cdot\vec E$")

sub1.plot(t,totWorkE_sim,color="r",label=r"$\vec v_{sim}\cdot\vec E+\vec v_{sim\ centered}\cdot\vec E$")

sub1.plot(t[1:],(energySim[1:]-energySim[:-1])/dt,color="k",linestyle="--",label=r"$\partial_t \mathbf{E}_{sim}$")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)




#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,I_totWorkE_sim,color="r",label=r"$\int\vec v_{sim}\cdot\vec E$")
sub1.plot(t,I_totWork_p_boris_ri,color="b",label=r"$\int\vec v_{boris\ ri}\cdot\vec E$")
sub1.plot(t,energySim,color="k",linestyle="--",label=r"$\mathbf{E}_{sim}$")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(I_totWorkE_sim-energySim)/energySim,label=r"$\vec v_{sim}\cdot\vec E$")
sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

"""
