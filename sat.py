#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:55:53 2022

@author: alexis
"""

#----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
vi = np.logspace(-3,0,200)[:-1]   #[c]
mi = np.array([32,64,100,1000,1836])
# mi = np.array([1836])  #[me]

n0 = 0.5
me = 1

gamma = 1/np.sqrt(1-vi**2)

wpe = np.sqrt(n0)  #c,e,eps0,me = 1
Te_ef  = n0*me*(gamma-1)   #[l_e^-3 me c^2]   electron temperature equal to initial drift kinetic energy

ratio        = np.zeros((len(mi),len(vi)))
ratio_upper  = np.zeros((len(mi),len(vi)))
ratio_lower  = np.zeros((len(mi),len(vi)))

for i in range(len(mi)):

    kin_i  = n0*mi[i]*(gamma-1)   #[l_e^-3 me c^2]   initial ion drift kinetic energy density
    wpi = wpe / np.sqrt(mi[i])  #units of [wpe]

    gammaFil = vi/np.sqrt(gamma) * wpi    #[wpe]
    kFil     = 1 /np.sqrt(gamma) * wpi    #[wpe/c]

    bsat = gamma*gammaFil**2*mi[i] / (vi*kFil*2*np.pi) #trapping

    alpha = 2/3
    gammaKink = 1/alpha * vi * wpi * np.sqrt(me/mi[i]) #[wpe], assuming R0 = c/wpi

    #from Faraday's law
    # Te_if = n0 * alpha/(2*np.pi) * vi**2 * mi[i] * (mi[i]/me)**(3/4)
    Te_if = bsat * vi**2 / gammaKink

    Te_if= (bsat*vi)**2/2*mi[i]/me

    epsB = bsat**2/2
    epsB_classical = 1/2 /(2*np.pi)**2 * n0*mi[i]*vi**2  #classical trapping = trapping/gamma

    #kumar 2015
    # p2 = 2*mi[i]*epsB/n0
    # e_ene = me*(np.sqrt(1+p2/me**2)-1)
    # Te_if = n0*e_ene



    max_Te_if = 1/4 * mi[i]**2/me * vi**2

    # Te_if_upper = np.sqrt(2*mi[i]*epsB/n0 * 2)
    # Te_if_lower = np.sqrt(2*mi[i]*epsB/n0 * 0.5)

    #should be independant of mass ratio
    ratio[i] = (Te_ef + Te_if) / (kin_i - Te_if)
    # ratio_upper[i] = (Te_ef + Te_if_upper) / (Ti_k - Te_if_upper)
    # ratio_lower[i] = (Te_ef + Te_if_lower) / (Ti_k - Te_if_lower)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axvline(0.5,color="gray",linestyle="--",linewidth=0.7)
sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)

for i in range(len(mi)):
    sub1.plot(vi, ratio[i],label=r"$m_i/m_e=%.0f$"%(mi[i]/me))
    # sub1.plot(vi, ratio_upper[i],label=r"$m_i/m_e=%.0f$"%(mi[i]/me))
    # sub1.plot(vi, ratio_lower[i],label=r"$m_i/m_e=%.0f$"%(mi[i]/me))
    # plt.fill_between(vi, ratio[i], ratio_lower[i], facecolor='gray', alpha=0.5)
    # plt.fill_between(vi, ratio[i], ratio_upper[i], facecolor='gray', alpha=0.5)

sub1.set_xlim(min(vi),max(vi))
sub1.set_xscale("log")
sub1.set_yscale("log")

sub1.set_xlabel(r"$v_i [c]$")
sub1.set_ylabel(r"$T_e/T_i$")

sub1.legend(frameon=False)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(vi,Te_ef,color="b",label=r"$T_e^{ef}$")
sub1.plot(vi,Te_if,color="g",label=r"$T_e^{if}$")
sub1.plot(vi,kin_i,color="r",label=r"$kin_i$")
sub1.plot(vi,epsB,color="k",label=r"$B_{if}^2/2\ (relativistic)$")
sub1.plot(vi,epsB_classical,color="cyan",linestyle="--",label=r"$B_{if}^2/2\ (classical)$")
sub1.plot(vi,max_Te_if,color="orange",linestyle="--",label=r"$\max T_e$")

sub1.set_xscale("log")
sub1.set_yscale("log")

sub1.legend(frameon=False)
