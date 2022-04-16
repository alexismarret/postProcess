#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:22:24 2022

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
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="counterStream"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,1)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis("eL")[st]

#----------------------------------------------
def cov(a,b):

    #naive implementation, might have issue because of machine precision
    # c = np.mean((a - np.mean(a,axis=(1,2))[...,None,None]) *
    #             (b - np.mean(b,axis=(1,2))[...,None,None]),
    #             axis=(1,2))

    #shifted implementation, more robust
    v = 1  #grid point to use, arbitrary value
    kx = a[:,v,v][:,None,None]
    ky = b[:,v,v][:,None,None]

    c = (np.mean((a-kx) * (b-ky),axis=(1,2)) -
         np.mean(a-kx,axis=(1,2)) * np.mean(b-ky,axis=(1,2)))

    return c


# cov_iL = cov(o.getUth(time,"iL","x")**2*o.getRatioQM("iL"),
#              o.getCharge(time, "iL"))

# cov_iR = cov(o.getUth(time,"iR","x")**2*o.getRatioQM("iR"),
#               o.getCharge(time,"iR"))

# cov_eL = cov(o.getUth(time,"eL","x")**2,
#               o.getCharge(time,"eL") *-1)

# cov_eR = cov(o.getUth(time,"eR","x")**2,
#               o.getCharge(time,"eR") *-1)
Ex =  (o.getE(time,"x"))
cov_iL = cov(o.getUth(time,"iL","x")**2*o.getRatioQM("iL"),
             Ex)

cov_iR = cov(o.getUth(time,"iR","x")**2*o.getRatioQM("iR"),
              Ex)

cov_eL = cov(o.getUth(time,"eL","x")**2,
              Ex)

cov_eR = cov(o.getUth(time,"eR","x")**2,
              Ex)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,cov_iL,color="r",label="iL")
sub1.plot(time,cov_iR,color="b",label="iR")

sub1.plot(time,cov_eL,color="g",label="eL")
sub1.plot(time,cov_eR,color="orange",label="eR")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$Cov(T(x,y),E(x,y))$")

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.legend(frameon=False)
