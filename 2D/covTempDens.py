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
import parallelFunctions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS2DrmhrTrack"
o = osiris.Osiris(run,spNorm="iL")

st = slice(None,None,1)
x    = o.getAxis("x")
y    = o.getAxis("y")
time = o.getTimeAxis()[st]

#----------------------------------------------
def cor(a,b):

    #naive implementation, might have issue because of machine precision
    # c = np.mean((a - np.mean(a,axis=(1,2))[...,None,None]) *
    #             (b - np.mean(b,axis=(1,2))[...,None,None]),
    #             axis=(1,2))

    #shifted implementation, more robust
    v = 1  #grid point to use, arbitrary value
    kx = a[:,v,v][:,None,None]
    ky = b[:,v,v][:,None,None]

    ax=(1,2)
    c = ((np.nanmean((a-kx) * (b-ky),axis=ax) -
         np.nanmean(a-kx,axis=ax) * np.nanmean(b-ky,axis=ax)) /
         (np.nanstd((a),axis=ax)*np.nanstd((b),axis=ax)))

    return c


"""
#----------------------------------------------
cor_iL = cor(o.getUth(time,"iL","x")**2*o.rqm[o.sIndex("iL")],
              o.getCharge(time, "iL"))

cor_iR = cor(o.getUth(time,"iR","x")**2*o.rqm[o.sIndex("iR")],
              o.getCharge(time,"iR"))

cor_eL = cor(o.getUth(time,"eL","x")**2,
              o.getCharge(time,"eL") *-1)

cor_eR = cor(o.getUth(time,"eR","x")**2,
              o.getCharge(time,"eR") *-1)

# eps = 1e-6   #avoid /0
# ratio = np.mean(np.abs(o.getUfluid(time, "iL","y") /
#                        (o.getUfluid(time, "iL","x")+eps)),axis=(1,2))

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,cor_iL,color="r",label="iL")
sub1.plot(time,cor_iR,color="b",label="iR")

sub1.plot(time,cor_eL,color="g",label="eL")
sub1.plot(time,cor_eR,color="orange",label="eR")

# sub1.plot(time,ratio,color="k",label="uy/ux")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$Cor(T(x,y),n(x,y))$")

# sub1.set_ylim(-0.0015,0.0015)
sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.legend(frameon=False)

plt.savefig(o.path+"/plots/correlation.png",dpi="figure")
"""

"""
Ex = o.getE(time,"x")
Ey = o.getE(time,"y")
Ez = o.getE(time,"z")

Bx = o.getB(time,"x")
By = o.getB(time,"y")
Bz = o.getB(time,"z")

skew = o.getNewData(time, "skew")

ExB_B2_x = o.crossProduct(Ex,Ey,Ez,
                            Bx,By,Bz)[0] / (Bx**2+By**2+Bz**2)

cor_exB_skew = cor(np.abs(ExB_B2_x, skew)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,cor_exB_skew,color="r",label="iL")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$Cor(T(x,y),n(x,y))$")

# sub1.set_ylim(-0.0015,0.0015)
sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.legend(frameon=False)

plt.savefig(o.path+"/plots/correlationExB.png",dpi="figure")
"""


#----------------------------------------------
skew = o.getNewData(time, "skew")
cov_iL = cor(o.getUth(time,"iL","x")**2*o.rqm[o.sIndex("iL")],
              skew)

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,cov_iL ,color="r",label="iL")

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub1.set_ylabel(r"$Cor(T(x,y),n(x,y))$")

# sub1.set_ylim(-0.0015,0.0015)
sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.legend(frameon=False)

plt.savefig(o.path+"/plots/correlationSkewTi.png",dpi="figure")



