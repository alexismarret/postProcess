#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:46:48 2022

@author: alexis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:09:24 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
# run  = ["CS3D_noKink","CS3Dtrack"]
# run = ["CS3Drmhr","CS3Drmhr"]
run = ["CS2DrmhrTrack"]
o = osiris.Osiris(run[0],spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)
# sl = (sx,sy,sz)
sl = (sx,sy)
# ax = (0,1,2)
ax = (0,1)

st = slice(None)
time = o.getTimeAxis()[st]

B2 = o.getEnergyIntegr(time, qty="B")[2]
#----------------------------------------------
#no kink
ENoKink  = np.zeros(len(time))
EcNoKink = np.zeros(len(time))
ErNoKink = np.zeros(len(time))

o = osiris.Osiris(run[0],spNorm="iL")

for i in range(len(time)):
    E  = o.getE(time[i], "x", sl=sl)
    Ec = o.getNewData(time[i], "Ecx", sl=sl)

    ENoKink[i]   = np.sqrt(np.mean(E**2,     axis=ax))
    EcNoKink[i]  = np.sqrt(np.mean(Ec**2,    axis=ax))
    ErNoKink[i]  = np.sqrt(np.mean((E-Ec)**2,axis=ax))

# #----------------------------------------------
# #kink
# EKink    = np.zeros(len(time))
# EcKink   = np.zeros(len(time))
# ErKink   = np.zeros(len(time))

# o = osiris.Osiris(run[1],spNorm="iL")

# for i in range(len(time)):
#     E  = o.getE(time[i], "x", sl=sl)
#     Ec = o.getNewData(time[i], "Ecx", sl=sl)

#     EKink[i]   = np.sqrt(np.mean(E**2,     axis=ax))
#     EcKink[i]  = np.sqrt(np.mean(Ec**2,    axis=ax))
#     ErKink[i]  = np.sqrt(np.mean((E-Ec)**2,axis=ax))


#%%
#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,B2,color="g")

# sub1.plot(time,EKink,label=r"$E_x\ L_x=45.2\ c/\omega_{pi}$",color="b")
# sub1.plot(time,EcKink,label=r"$E_{cx}\ L_x=45.2\ c/\omega_{pi}$",color="b",linestyle="dashed")
# sub1.plot(time,ErKink,label=r"$E_{rx}\ L_x=45.2\ c/\omega_{pi}$",color="b",linestyle="dotted")

sub1.plot(time,ENoKink,label=r"$E_x\ L_x=11.2\ c/\omega_{pi}$",color="r")
sub1.plot(time,EcNoKink,label=r"$E_{cx}\ L_x=11.2\ c/\omega_{pi}$",color="r",linestyle="dashed")
sub1.plot(time,ErNoKink,label=r"$E_{rx}\ L_x=11.2\ c/\omega_{pi}$",color="r",linestyle="dotted")

sub1.set_xlabel(r'$t\ [\omega_{pi}^{-1}]$')
sub1.set_ylabel(r'$<E_x^2>^{1/2}\ [E_0]$')

sub1.legend(frameon=False)
