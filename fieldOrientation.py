#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:27:52 2022

@author: alexis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:26:32 2022

@author: alexis
"""


#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import parallel_functions as pf
from scipy import signal

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True, 'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="counterStreamFast"
o = osiris.Osiris(run,spNorm="iL")

x     = o.getAxis("x")
y     = o.getAxis("y")          #18-22
time = o.getTimeAxis("eL")

xpos = 0
t = 1
time = time[t]
filtr = False

#----------------------------------------------
Bx = o.getB(time,"x")[0,xpos]
By = o.getB(time,"y")[0,xpos]
Bz = o.getB(time,"z")[0,xpos]

Ex = o.getE(time,"x")[0,xpos]
Ey = o.getE(time,"y")[0,xpos]
Ez = o.getE(time,"z")[0,xpos]

j_iLx = (o.getCurrent(time, "eL", "x")[0,xpos]+
         o.getCurrent(time, "eR", "x")[0,xpos]+
         o.getCurrent(time, "iL", "x")[0,xpos]+
         o.getCurrent(time, "iR", "x")[0,xpos])

if filtr:
    st=1.     #standard deviation for gaussian filter
    win=signal.gaussian(len(y),st)   #window

    Bx      = signal.convolve(Bx, win, mode='same') / np.sum(win)
    By      = signal.convolve(By, win, mode='same') / np.sum(win)
    Bz      = signal.convolve(Bz, win, mode='same') / np.sum(win)

    Ex      = signal.convolve(Ex, win, mode='same') / np.sum(win)
    Ey      = signal.convolve(Ey, win, mode='same') / np.sum(win)
    Ez      = signal.convolve(Ez, win, mode='same') / np.sum(win)

    j_iLx   = signal.convolve(j_iLx, win, mode='same') / np.sum(win)


angle_B=np.arctan2(Bz,Bx)
# phi_Bj = (2*np.pi + angle_Bj)*(angle_Bj<0) + angle_Bj*(angle_Bj>=0) #rescale between 0 and 2*pi

phi_B=angle_B

angle_E=np.arctan2(Ey,Ex)
# phi_BE = (2*np.pi + angle_BE)*(angle_BE<0) + angle_BE*(angle_BE>=0)
phi_E = angle_E


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,1.8),dpi=300)

def skip_jump(sub, axis, phi, max_jump, color):

    #----------------------------------------------
    #don't draw large change over one grid point in phase
    #max_jump is maximum phase change over one grid point allowed to be displayed

    skip=[0]
    for k in range(len(axis)-1):
        if np.abs(phi[k]-phi[k+1]) >= max_jump:
            skip.append(k+1)

    for sk in range(len(skip)-1) :
        sl = slice(skip[sk],skip[sk+1])
        sub.plot(axis[sl], phi[sl], color=color)

    #plot last segment
    try:
        sub.plot(axis[skip[sk+1]:], phi[skip[sk+1]:], color=color)
    #plot everything if no skips
    except UnboundLocalError:
        sub.plot(axis, phi, color=color)

    return

sub1.axhline(-np.pi/2,  color="gray",linestyle="--",linewidth=0.7)
sub1.axhline(np.pi/2,color="gray",linestyle="--",linewidth=0.7)

skip_jump(sub1, y, phi_B, 2*np.pi,"r")
skip_jump(sub1, y, phi_E, 2*np.pi,"b")



#----------------------------------------------
# sub1.set_ylim(0,2*np.pi)


sub1.set_xlabel(r'$y\ [l_0]$')

# sub1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
# sub1.set_yticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

sub1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
sub1.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

# sub1.legend(frameon=False)

