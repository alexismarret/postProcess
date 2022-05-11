#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:32:41 2022

@author: alexis
"""


import osiris
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run = "CS2Drmhr"

o = osiris.Osiris(run,spNorm="eL")

#----------------------------------------------
st = slice(None,None,1)
time = o.getTimeAxis(ene=True)[st]

En_B = o.getEnergyIntegr(time, "B")
En_E = o.getEnergyIntegr(time, "E")

kin_el = o.getEnergyIntegr(time, qty="kin", species="eL")
kin_er = o.getEnergyIntegr(time, qty="kin", species="eR")
kin_il = o.getEnergyIntegr(time, qty="kin", species="iL")
kin_ir = o.getEnergyIntegr(time, qty="kin", species="iR")

# U_int_iL =  np.mean(o.getCharge(time,"iL")*(o.getUth(time, "iL", "x")**2 +
#                                             o.getUth(time, "iL", "y")**2 +
#                                             o.getUth(time, "iL", "z")**2),
#                                             axis=(1,2))/2*o.getRatioQM("iL")

# U_int_iR =  np.mean(o.getCharge(time,"iR")*(o.getUth(time, "iR", "x")**2 +
#                                             o.getUth(time, "iR", "y")**2 +
#                                             o.getUth(time, "iR", "z")**2),
#                                             axis=(1,2))/2*o.getRatioQM("iR")

# U_int_eL = -np.mean(o.getCharge(time,"eL")*(o.getUth(time, "eL", "x")**2 +
#                                             o.getUth(time, "eL", "y")**2 +
#                                             o.getUth(time, "eL", "z")**2),
#                                             axis=(1,2))/2

# U_int_eR = -np.mean(o.getCharge(time,"eR")*(o.getUth(time, "eR", "x")**2 +
#                                             o.getUth(time, "eR", "y")**2 +
#                                             o.getUth(time, "eR", "z")**2),
#                                             axis=(1,2))/2

# E_tot = En_E+En_B+kin_el+kin_il+kin_er+kin_ir+U_int_iL+U_int_iR+U_int_eL+U_int_eR
E_tot = (np.sum(En_E,axis=-1)/2 +
         np.sum(En_B,axis=-1)/2 +
         kin_el + kin_il + kin_er + kin_ir)


#----------------------------------------------
fig, ((sub1,sub2),(sub3,sub4)) = plt.subplots(2,2,figsize=(4.1,1.8),dpi=300)

sub1.semilogy(time,kin_el,label=r"$kin\ el$")
sub1.semilogy(time,kin_il,label=r"$kin\ il$")
sub2.semilogy(time,kin_er,label=r"$kin\ er$")
sub2.semilogy(time,kin_ir,label=r"$kin\ ir$")

sub3.semilogy(time,En_E[:,0],label=r"$E_x$")
sub3.semilogy(time,En_E[:,1],label=r"$E_y$")
sub3.semilogy(time,En_E[:,2],label=r"$E_z$")

sub4.semilogy(time,En_B[:,0],label=r"$B_x$")
sub4.semilogy(time,En_B[:,1],label=r"$B_y$")
sub4.semilogy(time,En_B[:,2],label=r"$B_z$")

sub1.legend(frameon=False)
sub2.legend(frameon=False)
sub3.legend(frameon=False)
sub4.legend(frameon=False)


"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,1.8),dpi=300)

sub1.semilogy(time,En_E)
sub1.semilogy(time,En_B)

sub1.semilogy(time,kin_el)
sub1.semilogy(time,kin_er)
sub1.semilogy(time,kin_il)
sub1.semilogy(time,kin_ir)

sub1.semilogy(time,U_int_eL)
sub1.semilogy(time,U_int_eR)
sub1.semilogy(time,U_int_iL)
sub1.semilogy(time,U_int_iR)

sub1.semilogy(time,E_tot)
"""
"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,1.8),dpi=300)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.plot(time,(E_tot-E_tot[0])/E_tot[0])

sub1.legend(frameon=False)
sub1.set_xlim(time[0],time[-1])
sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$(\mathcal{E}-\mathcal{E}_0)/\mathcal{E}_0$")
"""
"""
the total energy reported by the scalar HIST diagnostics are:

    Particle energies: mass (rqm) * density * kinetic energy (gamma-1) * number of skin depths in the simulation

    (in 2D and 3D, this should be the number of skin depths squared or cubed, i.e. xmax*ymax*zmax in 3D)

    Field energies: field^2/2 * number of skin depths in the simulation

    (in 2D and 3D, this should be the number of skin depths squared or cubed, i.e. xmax*ymax*zmax in 3D).
    Note that in normalized units where the distances are units of skin depths,
    this energy is the same as the particle energy, i.e. you should not use E1^2/8pi while you are in normalized units.

So in 1D, this report gives you energy/area, since you aren't integrating over the missing axes,
in 2D it gives energy/distance, and in 3D just an energy.
The energy densities (energy/cubic skin depths) resolved in 1D, 2D, or 3D can be viewed
by adding "ene_e", or "ene_b" to the reports element in the diag_emf section,
or by adding "ene" to the reports element of any diag_species section.

    Particle energies (ene): mass (rqm) * density * kinetic energy (gamma-1)

    Field energies (ene_b and ene_e): field^2.

    NOTE: This is NOT the same as in the HIST section.
    In fact this is wrong, it is double the true energy density in normalized units.
    The true energy density of the field components should be field^2/2 in normalized units, not just field^2.
    This caused me a lot of issues before I realized it was implemented differently than in the scalar diagnostics.
    So if you have a plasma where ene_b is the same as the particle ene diagnostic,
    then it actually has twice as much kinetic energy density as magnetic energy
    because the ene_b and ene_e diagnostics over-report the energy density by a factor of 2.

"""
