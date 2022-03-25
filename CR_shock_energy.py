#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:06:57 2022

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
#cosmic rays energy fraction
#============================
plt.close("all")

M0 = np.logspace(0,2,1000)
mu = 1. - np.logspace(-4,0,10)[1:]

gam = 5/3.
me_mp = 1./1836

#============================
# r1 = ((gam+1)*M0**2) / ((gam-1)*M0**2+2)
r1 = ((gam-1)*M0**2/(2*gam))**(1./(gam+1)) #gives maximum r1*r2, supported by kinetic modeling

P1th_P0 = r1**gam

M1 = M0*np.sqrt(1./(P1th_P0*r1))

r2 = ((gam+1)*M1**2) / ((gam-1)*M1**2+2)

alp = 1 + 2./((gam-1)*M0**2) - 1./(r1*r2)* (2./((gam-1)*M0**2) + 2.*gam/(gam-1)*(1-1./(r1*r2)) + 1./(r1*r2))


P2_P0 = 1+gam*M0**2*(1-1./(r1*r2))

P2th_P0 = r1**gam +gam*M0**2*(1./r1-1./(r1*r2))

P2cr_P0 = 1-r1**gam+gam*M0**2*(1-1./r1)

P1e_P0 = 1+gam*M0[...,None]**2-gam*M0[...,None]**2*me_mp*(1-mu[None,...])*((gam-1)/(2*gam)*M0[...,None]**2)**(gam+1)

#============================
gam1i = 100.
gam1e = 10.

c = 1.

v1i = c*np.sqrt(1-1./gam1i**2)
v1e = c*np.sqrt(1-1./gam1e**2)

v1i_prime = (v1i-v1e)/(1-v1i*v1e/c**2)
gam1i_prime = 1./np.sqrt(1-v1i_prime**2/c**2)

mu = 1. - v1e/v1i
n1e_n1i = 1./(1-mu)
Li = (1-mu)*np.sqrt(2*(gam1i_prime-1)/mu)


print(v1i/c,v1e/c,v1i_prime/c,gam1i_prime,mu,Li)

import sys
sys.exit()
#============================
fig, (sub1) = plt.subplots(1,figsize=(4.1,2),dpi=300,sharex=True)

sub1.axhline(0,color="gray",linestyle="--")



sub1.loglog(M0,r1-1,color="k")
"""
cond = M0/(r1**((gam+1)/2.)) >= 1.
cond[0]=False

sub1.loglog(M0[cond],alp[cond],color="b")
sub1.loglog(M0[~cond],alp[~cond],color="b")



sub1.loglog(M0,P2_P0,color="orange")

sub1.loglog(M0,P2th_P0,color="k")
sub1.loglog(M0,P2cr_P0,color="k",linestyle="--")

sub1.loglog(M0,M0)
"""
sub1.loglog(M0,P1e_P0)

sub1.set_xlabel(r"$M_0$")
sub1.set_ylabel(r"$\alpha$")


sub1.set_yscale("linear")

sub1.set_xlim(1,100)
sub1.set_ylim(-1,50)

