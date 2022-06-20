#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:23:50 2022

@author: alexis
"""

# ----------------------------------------------
import osiris
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# from matplotlib.gridspec import GridSpec
# import trackParticles as tr
import time as ti

# ----------------------------------------------
params = {
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "lines.linewidth": 1,
    "lines.markersize": 3,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 9,
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.1,
    "legend.labelspacing": 0.1,
    "axes.linewidth": 1,
    "figure.autolayout": True,
    "text.usetex": True,
}
plt.rcParams.update(params)
plt.close("all")

# ----------------------------------------------
run = "CS3Dtrack"
# run="testTrackSingle"
spNorm = "eL"
o = osiris.Osiris(run, spNorm=spNorm)

species = "eL"

# ----------------------------------------------
# index of macroparticle
sPart = slice(None, None, 1000)
sTime = slice(None, None, 1)
sl = (sPart, sTime)

# ----------------------------------------------
muNorm = o.rqm[o.sIndex(spNorm)]
mu = o.rqm[o.sIndex(species)]
pCharge = mu / np.abs(mu)

# macroparticle, iteration
ene = o.getTrackData(species, "ene", sl=sl) * np.abs(mu)
t = o.getTrackData(species, "t", sl=sl)
p1 = o.getTrackData(species, "p1", sl=sl)
p2 = o.getTrackData(species, "p2", sl=sl)
p3 = o.getTrackData(species, "p3", sl=sl)
e1 = o.getTrackData(species, "E1", sl=sl)
e2 = o.getTrackData(species, "E2", sl=sl)
e3 = o.getTrackData(species, "E3", sl=sl)
b1 = o.getTrackData(species, "B1", sl=sl)
b2 = o.getTrackData(species, "B2", sl=sl)
b3 = o.getTrackData(species, "B3", sl=sl)

# in osiris: track data in file os-spec-diagnostics.f03 line 1886
# also in os-spec-tracks.f03 line 829
# os-vdf-interpolate.f90 line 700

N = len(t[0])  # number of time steps
dt = t[0, 1] - t[0, 0]  # time step

imax = np.where(ene[:, -1] == np.max(ene[:, -1]))[0][
    0
]  # index of most energetic particle

# ----------------------------------------------
def cross(a, b):
    return np.array(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            -a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ]
    ).T


def lorentz(p):
    return np.sqrt(1 + np.sum(p ** 2, axis=1))


def getEnergy(p, mu):
    return (lorentz(p) - 1) * mu


# ----------------------------------------------
E = np.array([e1, e2, e3]).T
B = np.array([b1, b2, b3]).T
p = np.array([p1, p2, p3]).T

t = t[0]
ene = ene[0]
E = E[:, 0]
B = B[:, 0]
p = p[:, 0]

lorentz_p = lorentz(p)[..., None]

# ----------------------------------------------
# Forward Euler
pnext_euler = p + pCharge * (E + cross(p / lorentz_p, B)) * dt
eneEuler = getEnergy(pnext_euler, mu)

# ----------------------------------------------
# Boris push
# half electric push
Epush = pCharge * E * dt / 2
pminus = p + Epush

# rotation
tvec = pCharge * B * dt / (2 * lorentz_p)
s = 2 * tvec / (1 + np.sum(tvec ** 2, axis=1)[:, None])
pprime = pminus + cross(pminus, tvec)
pplus = pminus + cross(pprime, s)

# second half electric push
pnext_boris = pplus + Epush

# ----------------------------------------------
energy_initial = getEnergy(p, mu)  # energy before push == ene array
energy_first_push = getEnergy(pminus, mu)  # energy after first push
energy_rot = getEnergy(pplus, mu)  # energy after rotation
energy_second_push = getEnergy(pnext_boris, mu)  # energy after second push
energy_gain = energy_second_push - energy_initial  # energy gain from initial value

# ----------------------------------------------
# Now look at the energy change by comparing the energy change at each step with the energy change in the simulation
dEdt_sim = (ene[1:] - ene[:-1]) / dt  # energy gained from the simulation data

dEdt_boris = (energy_second_push[1:] - energy_second_push[:-1]) / dt
dEdt_euler = (eneEuler[1:] - eneEuler[:-1]) / dt

workE_basic = np.sum(E * p / lorentz_p, axis=1)[:-1] * pCharge

# ----------------------------------------------
lorentz_pplus = lorentz(pplus)[:, None]
workFP = (
    np.sum(E * dt / 2 * (p / lorentz_p), axis=1) * pCharge
)  # energy gain via E in first push
workSP = (
    np.sum(E * dt / 2 * (pplus / lorentz_pplus), axis=1) * pCharge
)  # energy gain via E in second push

workE_boris = (
    workFP + workSP
)  # total energy gain via E, obtained from boris velocity, slightly different from sim velocity
"""
#----------------------------------------------
plt.figure(figsize=(4.1,2.8),dpi=300)

#first and second push give slightly different energies, since v varies every half time step
#because of boris acceleration in two steps
plt.plot(t,(energy_first_push-energy_initial),color="b")
plt.plot(t,(energy_rot-energy_first_push),color="k")
plt.plot(t,(energy_second_push-energy_first_push),color="r")

plt.plot(t,(energy_gain),color="g",linestyle="-")
plt.plot(t[1:]-dt,dEdt_sim*dt)

plt.plot(t,workFP,color="cyan",linestyle="--")
plt.plot(t,workSP,color="k",linestyle="--")
plt.plot(t,workE_boris,color="orange",linestyle="--")
"""
"""
#----------------------------------------------
#now get fraction of energy gain due to each component
#do not integrate work itself, since accumulation of error makes the result wrong
workEx = E[:,0] * (p[:,0]/lorentz_p + pplus[:,0]/lorentz_pplus) * pCharge*dt/2
workEy = E[:,1] * (p[:,1]/lorentz_p + pplus[:,1]/lorentz_pplus) * pCharge*dt/2
workEz = E[:,2] * (p[:,2]/lorentz_p + pplus[:,2]/lorentz_pplus) * pCharge*dt/2
"""
"""
#----------------------------------------------
plt.figure(figsize=(4.1,2.8),dpi=300)

plt.plot(t,workEx,color="r")
plt.plot(t,workEy,color="g")
plt.plot(t,workEz,color="b")

plt.plot(t,workE,color="k")
plt.plot(t,workEx+workEy+workEz,color="cyan",linestyle="--")
"""
"""
#----------------------------------------------
plt.figure(figsize=(4.1,2.8),dpi=300)

plt.plot(t,np.cumsum(workEx))
plt.plot(t,np.cumsum(workEy))
plt.plot(t,np.cumsum(workEz))
"""

# ----------------------------------------------
plt.figure(figsize=(4.1, 2.8), dpi=300)

plt.plot(t[1:] - dt / 2, dEdt_sim, label="sim", linestyle="-")
plt.plot(
    t[1:] - dt / 2,
    workE_basic,
    label=r"$\mathbf{v} \cdot \mathbf{E}_{basics}$",
    linestyle="-",
)
plt.plot(
    t - dt / 2,
    workE_boris,
    label=r"$\mathbf{v} \cdot \mathbf{E}_{boris}$",
    linestyle="-",
)
plt.plot(t[1:] + dt / 2, dEdt_euler, label="Forward Euler", linestyle="-")
plt.plot(t[1:] + dt / 2, dEdt_boris, label="Boris", linestyle="--")

plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"Work/dt")
plt.show()


"""
params = [
    [dEdt_sim,'sim','-'],
    [WorkE,r'$\mathbf{v} \cdot \mathbf{E}$','-'],
    [dEdt_euler,'Forward Euler','-'],
    [dEdt_boris,'Boris','--'],
]

plt.figure(figsize=(4.1,2.8),dpi=300)
for dat,label,ls in params:
    plt.plot(t[1:],np.cumsum(dat)*dt+ene[0],ls,label=label)
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'Energy')
plt.show()


plt.figure(figsize=(4.1,2.8),dpi=300)
for dat,label,ls in params:
    plt.semilogy(t[1:],np.abs(np.cumsum(dat)*dt+ene[0] - ene[1:])/ene[1:],ls,label=label)
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'Energy error')
plt.show()

"""
