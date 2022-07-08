#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:57:02 2022

@author: alexis
"""

import osiris
import numpy
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
# plt.close("all")

#----------------------------------------------
run  ="CS3D_noKink"
# run = 'CS2DrmhrTrack'

o = osiris.Osiris(run)

sx = slice(None)
sy = slice(None)
sz = slice(None)
sl = (sx,sy,sz)

x = o.getAxis("x")[sx]

# st = slice(None,None,1)
st = 10
time = o.getTimeAxis()[st]


#multiple of the mesh size
#signal on scales < alpha*meshSize will be filtered out
#must be larger that 2 because of Nyquist theorem
alpha = 4
cutoff = 1/(alpha*o.meshSize)

#----------------------------------------------
data = o.getE(time, "x", sl=sl)
o.low_pass_filter(data, cutoff=cutoff, axis=(1,2,3))




"""

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(x,data)
sub1.plot(x,filtered_data)

sub1.set_xlabel(r'$t\ [\omega_{pi}^{-1}]$')
sub1.set_ylabel(r'$<E^2>\ [E_0^2]$')


"""
