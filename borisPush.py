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

    nrgy_push1 = energy(pm,mass)

    #get p_prime
    tvec = fac / lorentz(pm)[...,None] * bf
    pprime = pm + cross(pm, tvec, it)

    #get p_plus
    svec = tvec * 2. / (1.+np.sum(tvec**2,axis=-1))[...,None]
    pp = pm + cross(pprime, svec, it)

    nrgy_rot = energy(pp,mass)

    #get next p
    pnp = pp + fac*ef

    nrgy_push2 = energy(pnp,mass)

    return pnp, pp, nrgy_push1, nrgy_rot, nrgy_push2


#----------------------------------------------
def energy(p, mass):

    return (lorentz(p)-1) * mass

#----------------------------------------------
def lorentz(p):

    return np.sqrt(1.+np.sum(p**2,axis=-1))


#----------------------------------------------
run ="CS3Dtrack"
# run="testTrackSingle"

spNorm = "eL"
o = osiris.Osiris(run,spNorm=spNorm)

species ="iL"

#----------------------------------------------
#index of macroparticle
ene = o.getTrackData(species, "ene")
# sPart = np.where(ene[:,-1]==np.max(ene[:,-1]))[0][0]  #index of most energetic particle
sPart = 30

sTime = slice(None,None,1)
sl = (sPart,sTime)

#----------------------------------------------
mass   = np.abs (o.rqm[o.sIndex(species)])
charge = np.sign(o.rqm[o.sIndex(species)])

#macroparticle, iteration
ene = ene[sl] * mass
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

energySim = ene

dendt      = (energySim[1:]   - energySim[:-1])/dt
# dendt = np.gradient(energySim,dt,edge_order=2)[:-1]

#----------------------------------------------
#only integrate next time step then come back to simulation data to integrate the next one
p_boris = np.zeros((N,3))
p_boris[0] = (p1[0],p2[0],p3[0])  #first value is initial condition (n-1/2 = -1/2)

#pp is momentum after first push and magnetic field rotation, first value is n=0, centered momentum
p_boris[1:], pp, nrgy_push1, nrgy_rot, nrgy_push2  = boris_relativistic(p[:-1], ef[:-1], bf[:-1],
                                                                        dt, charge, mass, it=False)

energyBoris = energy(p_boris, mass)   #energy after the two half pushes, starting from PIC data = nrgy_push2

#----------------------------------------------
#work from sim trajectory
workE_p  = p   /lorentz(p)  [...,None] *ef*charge  #work first push
workE_pp = pp  /lorentz(pp) [...,None] *ef[:-1]*charge  #work second push, after magnetic field rotation

#centered velocity work
workE_c = (p[:-1]/lorentz(p[:-1])[...,None] + p[1:]/lorentz(p[1:])[...,None]) * charge*ef[:-1]/2
I_workE_c = np.cumsum(np.sum(workE_c,axis=-1)*dt)+energySim[0]

I_workE_p  = np.sum(workE_p,axis=-1)  *dt/2
I_workE_pp = np.sum(workE_pp,axis=-1) *dt/2

#sum over two half pushes = 2*dE/dt and integral to get work to next time step from PIC data
incr = np.sum(workE_p[:-1] + workE_pp, axis=-1)*dt/2
#plus energy of previous step gives next energy
I_totWorkE_sim = incr + energySim[:-1]

#----------------------------------------------
deltaEnPush1 = (nrgy_push1-energySim[:-1])
deltaEnPush2 = (nrgy_push2-nrgy_push1)
errW_half1 = np.abs(I_workE_p[:-1]     -deltaEnPush1) / np.abs(deltaEnPush1)
errW_half2 = np.abs(I_workE_pp    -deltaEnPush2) / np.abs(deltaEnPush2)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:],I_workE_c,color="b")
# sub1.plot(t[:-1],(np.cumsum(workE_c)+energySim[0] - energySim[1:])/energySim[1:])
sub1.plot(t,energySim,color="r")

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[:-1],errW_half1,color="b",label=r"$(W_{push1}-\Delta\epsilon_{push1})/\Delta\epsilon_{push1}$")
          # label=r"$(W_E^n - \Delta\mathcal{E}_{boris}^n)/\Delta\mathcal{E}_{boris}^n$")

sub1.plot(t[:-1],errW_half2,color="r",linestyle="--",label=r"$(W_{push2}-\Delta\epsilon_{push2})/\Delta\epsilon_{push2}$")
          # label=r"$(W_E^n - \Delta\mathcal{E}_{boris}^n)/\Delta\mathcal{E}_{boris}^n$")

sub1.set_yscale("log")
sub1.legend(frameon=False)
plt.xlabel(r'$t$')


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,energyBoris,color="r",marker="o",label=r"$\mathcal{E}_{boris}$")
sub1.plot(t,energySim  ,color="k",linestyle="--",marker="x",label=r"$\mathcal{E}_{PIC}$")
sub1.plot(t[1:],I_totWorkE_sim,color="orange",marker="x",label=r"$\mathcal{E}_{PIC}^{n-1}+W_{boris}$")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(energyBoris-energySim)/energySim,
          label=r"$(\mathcal{E}_{boris}^n-\mathcal{E}_{PIC}^n)/\mathcal{E}_{PIC}^n$")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

"""
#----------------------------------------------
workE = charge * (p/lorentz(p)[:,None] ) *ef

I_workE = np.zeros(N)
I_workE[1:] = energySim[:-1]+np.sum(workE[:-1],axis=-1)*dt

# I_workE = cumulative_trapezoid(np.sum(workE,axis=-1),dx=dt,initial=0)+energySim[0]
"""
#----------------------------------------------
# epsilon_n_boris = np.cumsum(energyBoris)
# epsilon_n_sim = np.cumsum(energySim)

# error = (epsilon_n_boris-epsilon_n_sim)/epsilon_n_sim

"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[:-1],nrgy_push1,color="r")
sub1.plot(t[:-1],nrgy_rot,color="g",linestyle="--")
sub1.plot(t[:-1],nrgy_push2,color="b")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
"""

"""
#----------------------------------------------
#center velocity (p^n-1/2 / gamma^n-1/2 + p^n+1/2 / gamma^n+1/2)/2, product with E^n
vc = (p[:-1]/lorentz(p[:-1])[...,None] +
      p[1:] /lorentz(p[1:]) [...,None])/2
workEvc = charge * vc*ef[:-1]
I_workEvc = cumulative_trapezoid(np.sum(workEvc,axis=-1),t[:-1],initial=0)+energySim[0]

#----------------------------------------------
#center E field, (E^n + E^n+1)/2, product with p^n+1/2
Ec = (ef[:-1] +
      ef[1:])/2
workEc = charge * p[1:]/lorentz(p[1:])[:,None]*Ec
I_workEc = cumulative_trapezoid(np.sum(workEc,axis=-1),dx=dt,initial=0)+energySim[0]
"""

"""
#----------------------------------------------
#work E sim centered
workE_simc = charge * (pp/lorentz(pp)[...,None]) * ef[:-1]
s=np.sum(workE_simc,axis=-1)

# I_workE_sim = cumulative_trapezoid(s,dx=dt,initial=0) + energySim[0]

I_workE_sim = s*dt+ energySim[0]



I_workE_simc = (I_workE_sim[:-1]+I_workE_sim[1:])/2
# I_workE_simc = cumsimp(s,dt)+energySim[1]
# I_workE_simc = integr(s,dt)

# test = trapezoid(s,dx=dt)+ energySim[0]


#----------------------------------------------
#integrate whole trajectory from initial conditions, reinject previous solution
p_boris_ri = np.zeros((N,3))
p_boris_ri[0] = (p1[0],p2[0],p3[0])

pp_ri = np.zeros((N-1,3))
for i in range(0,N-1):
    p_boris_ri[i+1], pp_ri[i] = boris_relativistic(p_boris_ri[i], ef[i], bf[i],
                                                   dt, charge, mass, it=True)
energyBoris_ri = energy(p_boris_ri, mass)

#----------------------------------------------
#work from sim trajectory
workE_p  = p   /lorentz(p)  [...,None] *ef*charge  #work first push
workE_pp = pp  /lorentz(pp) [...,None] *ef[:-1]*charge  #work second push

totWorkE_sim = workE_p[:-1]+workE_pp   #sum over two half pushes

#divide by two since half time step each
I_WorkE_simX = cumulative_trapezoid(totWorkE_sim[:,0],dx=dt,initial=0)/2 + energySim[0]
I_WorkE_simY = cumulative_trapezoid(totWorkE_sim[:,1],dx=dt,initial=0)/2 + energySim[0]
I_WorkE_simZ = cumulative_trapezoid(totWorkE_sim[:,2],dx=dt,initial=0)/2 + energySim[0]
I_totWorkE_sim = cumulative_trapezoid(np.sum(totWorkE_sim,axis=-1),dx=dt,initial=0)/2 + energySim[0]

#----------------------------------------------
#work from reintegrated trajectory
workE_p_boris_ri  = p_boris_ri/lorentz(p_boris_ri) [...,None] *ef*charge  #work first push
workE_pp_ri = pp_ri/lorentz(pp_ri)[...,None] *ef[:-1]*charge #work second push

totWork_p_boris_ri = np.sum(workE_p_boris_ri[:-1]+workE_pp_ri,axis=-1)

I_totWork_p_boris_ri = cumulative_trapezoid(totWork_p_boris_ri,dx=dt,initial=0)/2 + energySim[0]
"""


"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[:-1],np.abs(I_workEvc-energySim[1:])/energySim[1:],
          label=r"$(\mathcal{E}_{borisvc\ next\ only}-\mathcal{E}_{sim})/\mathcal{E}_{sim}$")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:],np.abs(I_workEc-energySim[1:])/energySim[1:],
          label=r"$(\mathcal{E}_{borisc\ next\ only}-\mathcal{E}_{sim})/\mathcal{E}_{sim}$")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.abs(energyBoris_ri-energySim)/energySim,
          label=r"$(\mathcal{E}_{boris\ reintegrated}-\mathcal{E}_{sim})/\mathcal{E}_{sim}$")

sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)
"""
"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,p_boris_ri[:,0],color="r",label=r"$p1_{boris\ ri}$")
sub1.plot(t,p_boris_ri[:,1],color="g",label=r"$p2_{boris\ ri}$")
sub1.plot(t,p_boris_ri[:,2],color="b",label=r"$p3_{boris\ ri}$")

sub1.plot(t,p1,color="k",linestyle="--",label=r"$p1_{sim}$")
sub1.plot(t,p2,color="orange",linestyle="--",label=r"$p2_{sim}$")
sub1.plot(t,p3,color="cyan",linestyle="--",label=r"$p3_{sim}$")

plt.xlabel(r'$t$')
sub1.legend(frameon=False)

"""


"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t,np.sum(workE_p,axis=-1),color="cyan",
          label=r"$q\vec v_{sim}\cdot\vec E$",marker="x")
sub1.plot(t[:-1],np.sum(workE_pp,axis=-1),color="g",linestyle="--",
          label=r"$q\vec v_{sim\ centered}\cdot\vec E$",marker="o")

sub1.plot(t[:-1],np.sum(totWorkE_sim,axis=-1),color="r",
          label=r"$q(\vec v_{sim}\cdot\vec E+\vec v_{sim\ centered}\cdot\vec E)$")

sub1.plot(t[:-1],dendt,color="k",linestyle="--",
          label=r"$\partial_t \mathcal{E}_{sim}$")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)
# sub1.set_xlim(-1.5,3)
# sub1.set_ylim(-0.1,1.2)
"""
"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

#solve:
#red should be = blue
#green should be = orange

# I_dendt = cumulative_trapezoid(dendt,dx=dt,initial=0) + energySim[0]
# ene_c = (I_dendt[1:] + I_dendt[:-1])/2

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)

# sub1.plot(t[1:-1],I_workE_simc,color="g",linestyle="-",
#           label=r"$q\int v_{centered}\cdot Edt$")

sub1.plot(t,I_workE,color="r",linestyle="-",
          label=r"$q\int v_{PIC}\cdot Edt$")

# sub1.plot(t[:-1],I_dendt,color="r",marker="x",linestyle="dashed",
#           label=r"$\int \partial_t \mathcal{E}_{sim}dt$")

sub1.plot(t,ene,color="b",linestyle="dashdot",
          label=r"$\mathcal{E}_{sim}$")

# sub1.plot(t[:-2]+dt/2,ene_c,color="orange",marker="o",linestyle="--",
#           label=r"$(q\int v_{centered}\cdot E)_{centered}$")
# sub1.plot(t,energyBoris,color="k",linestyle="-",
#           label=r"$\mathcal{E}_{boris\ next}$")

plt.xlabel(r'$t$')
sub1.legend(frameon=False)
# sub1.set_xlim(-0.1,0.4)
# sub1.set_ylim(-0.01,0.04)
"""
"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:-1],np.abs(I_workE_simc-ene[1:-1])/ene[1:-1],
          label=r"$(\mathcal{E}_{sim\ centered}-\mathcal{E}_{sim})/\mathcal{E}_{sim}$")

plt.xlabel(r'$t$')
sub1.legend(frameon=False)
sub1.set_yscale("log")


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:],I_WorkE_simX,color="r",label=r"$q\int\vec v_xE_xdt$")
sub1.plot(t[1:],I_WorkE_simY,color="g",label=r"$q\int\vec v_yE_ydt$")
sub1.plot(t[1:],I_WorkE_simZ,color="b",label=r"$q\int\vec v_zE_zdt$")

plt.xlabel(r'$t$')
sub1.legend(frameon=False)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:],I_totWorkE_sim,color="r",label=r"$q\int\vec v_{sim}\cdot\vec Edt$")
sub1.plot(t[:-1],I_totWork_p_boris_ri,color="b",linestyle="--",label=r"$q\int\vec v_{boris\ ri}\cdot\vec Edt$")
sub1.plot(t,energySim,color="g",label=r"$\mathcal{E}_{sim}$")
sub1.plot(t,I_workE,color="cyan",label=r"$q\int\vec v_{sim\ no\ pp}\cdot\vec Edt$")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(t[1:],np.abs(I_totWorkE_sim-energySim[:-1])/energySim[:-1],
          label=r"$(q\int\vec v_{sim}\cdot\vec Edt-\mathcal{E}_{sim})/\mathcal{E}_{sim}$")
sub1.plot(t[:-1],np.abs(I_totWork_p_boris_ri-energySim[:-1])/energySim[:-1],
          label=r"$(q\int\vec v_{boris\ ri}\cdot\vec Edt-\mathcal{E}_{sim})/\mathcal{E}_{sim}$")

sub1.plot(t[1:],np.abs(I_totWork_p_boris_ri-energyBoris_ri[:-1])/energyBoris_ri[:-1],
          label=r"$(q\int\vec v_{boris\ ri}\cdot\vec Edt-\mathcal{E}_{boris\ ri})/\mathcal{E}_{boris\ ri}$")


sub1.set_yscale("log")
plt.xlabel(r'$t$')
sub1.legend(frameon=False)

"""
