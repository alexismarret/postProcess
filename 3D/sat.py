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
vi = np.logspace(-3,0,200)   #[c]
mi = np.array([32,64,100,1000,1836])
# mi = np.array([1836])  #[me]

n0 = 0.5
me = 1

gamma = 1/np.sqrt(1-vi**2)

wpe = np.sqrt(n0)  #c,e,eps0,me = 1
Te_ef  = n0*me   *(gamma-1)   #me c^2   electron temperature equal to initial drift kinetic energy

ratio        = np.zeros((len(mi),len(vi)))
ratio_upper  = np.zeros((len(mi),len(vi)))
ratio_lower  = np.zeros((len(mi),len(vi)))

for i in range(len(mi)):

    kin_i  = n0*mi[i]*(gamma-1)   #me c^2   initial ion drift kinetic energy density
    wpi = wpe / np.sqrt(mi[i])  #[wpe]

    #relativistic prediction of B filamentation
    gammaFil = vi/np.sqrt(gamma) * wpi    #[wpe]
    kFil     = 1 /np.sqrt(gamma) * wpi    #[wpe/c]


    #*2 because two components of B [le^-3 me c^2]


    bsat = gamma*mi[i]*gammaFil**2 / (kFil*vi) / np.sqrt(mi[i])**2 * np.sqrt(4*np.pi)
    epsB = bsat**2/2 * 2



    p2 = 2*epsB*mi[i]/n0
    # Te_if = n0*(np.sqrt(1+p2/mi[i]**2)-1)*mi[i]

    Te_if = n0*np.sqrt(2*mi[i]*epsB/n0)

    # Te_if_upper = np.sqrt(2*mi[i]*epsB/n0 * 2)
    # Te_if_lower = np.sqrt(2*mi[i]*epsB/n0 * 0.5)

    ratio[i]       = (Te_ef + Te_if) / (kin_i - Te_if)
    # ratio_upper[i] = (Te_ef + Te_if_upper) / (Ti_k - Te_if_upper)
    # ratio_lower[i] = (Te_ef + Te_if_lower) / (Ti_k - Te_if_lower)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axvline(0.5,color="gray",linestyle="--",linewidth=0.7)
# sub1.axhline(me/mi,color="gray",linestyle="--",linewidth=0.7)

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


sub1.plot(vi,Te_ef,color="r")
sub1.plot(vi,Te_if,color="g")
sub1.plot(vi,kin_i,color="b")

sub1.set_xscale("log")
sub1.set_yscale("log")

