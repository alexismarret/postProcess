#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:06:38 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt
import fit

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="counterStream6"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,1)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis()[st]

#----------------------------------------------
UiL = np.mean(o.getUfluid(time, "iL", "x"),axis=(1,2))
UiR = np.mean(o.getUfluid(time, "iR", "x"),axis=(1,2))

UeL = np.mean(o.getUfluid(time, "eL", "x"),axis=(1,2))
UeR = np.mean(o.getUfluid(time, "eR", "x"),axis=(1,2))

TiL = np.mean(o.getUth(time, "iL", "x")**2,axis=(1,2)) * o.getRatioQM("iL")[0]
TiR = np.mean(o.getUth(time, "iR", "x")**2,axis=(1,2)) * o.getRatioQM("iR")[0]

TeL = np.mean(o.getUth(time, "eL", "x")**2,axis=(1,2))
TeR = np.mean(o.getUth(time, "eR", "x")**2,axis=(1,2))

normB = np.mean(np.sqrt(o.getB(time,"x")**2+
                        o.getB(time,"y")**2+
                        o.getB(time,"z")**2),axis=(1,2))

GRavwB = np.gradient(normB)

sat = []
for i in range(len(GRavwB)-1):
    if len(sat)<3 and GRavwB[i+1]<0 and GRavwB[i]>0: sat.append(i)

# sl_ew = slice(min_ew,max_ew)
# sl_iw = slice(min_iw,max_iw)
# amp_ew, index_ew, rsquared_ew = fit.fitExponential(time[sl_ew], avwB[sl_ew])
# amp_iw, index_iw, rsquared_iw = fit.fitExponential(time[sl_iw], avwB[sl_iw])

#----------------------------------------------
fig, (sub1,sub2) = plt.subplots(1,2,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

sub1.semilogy(time,normB,color="g")
sub2.semilogy(time,normB,color="g")

sub1.semilogy(time,UiL,color="r")
sub1.semilogy(time,np.abs(UiR),color="k",linestyle="--")

sub2.semilogy(time,UeL,color="b")
sub2.semilogy(time,np.abs(UeR),color="k",linestyle="--")

sub1.semilogy(time,TiL,color="orange")
sub2.semilogy(time,TiR,color="orange")

sub1.semilogy(time,TeL,color="orange",linestyle="--")
sub2.semilogy(time,TeR,color="orange",linestyle="--")

for l in sat:
    sub1.axvline(time[l],color="k",linestyle="--",linewidth=0.7)
    sub2.axvline(time[l],color="k",linestyle="--",linewidth=0.7)

sub1.set_ylim(1e-6,2e-1)
sub1.set_xlim(time[0],time[-1])

sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")
sub2.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")

sub1.set_title(r"$Protons$")
sub2.set_title(r"$Electrons$")

# sub1.semilogy(time[sl_ew],amp_ew*np.exp(index_ew*time[sl_ew]),color="k")

# sub1.semilogy(time[sl_iw],amp_iw*np.exp(index_iw*time[sl_iw]),color="k")

# r = index_ew/index_iw
