#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:37:03 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)

#----------------------------------------------
# run  = ["CS3D_noKink","CS3Dtrack"]
# run = ["CS3Drmhr","CS3Drmhr"]
run = ["CS2DrmhrTrack"]
# run = ["Test3D/test3DdumpRaw"]
o = osiris.Osiris(run[0],spNorm="iL")

sx = slice(None,None,1)
sy = slice(None,None,1)
sz = slice(None,None,1)

y = o.getAxis("y")[sx]

# sl = (sx,sy,sz)
sl = (sx,sy)

st = slice(None,20,1)
time = o.getTimeAxis()[st]
timeSeries = True

#----------------------------------------------
Bx = o.getB(time, "x", sl=sl)
By = o.getB(time, "y", sl=sl)
Bz = o.getB(time, "z", sl=sl)

Ex  = o.getE(time, "x", sl=sl)
Ey  = o.getE(time, "y", sl=sl)
Ez  = o.getE(time, "z", sl=sl)

Ecx = o.getNewData(time, "Ecx", sl=sl)
Ecy = o.getNewData(time, "Ecy", sl=sl)
Ecz = o.getNewData(time, "Ecz", sl=sl)

Erx = Ex-Ecx
Ery = Ey-Ecy
Erz = Ez-Ecz

#----------------------------------------------
#dbdt = -curl E
ax = np.array([-1,-2,-3])[range(o.ndim)]

if timeSeries: dimkx = (1,o.grid[0]) + (1,)*(o.ndim-1)
else:          dimkx = (o.grid[0],)  + (1,)*(o.ndim-1)
kx = np.fft.fftfreq(o.grid[0],o.meshSize[0]) .reshape(dimkx)
if o.ndim>1:  #2D
    if timeSeries: dimky = (1,o.grid[1]) + (1,)*(o.ndim-2)
    else:          dimky = (o.grid[1],)  + (1,)*(o.ndim-2)
    ky = np.fft.fftfreq(o.grid[1],o.meshSize[1]) .reshape(dimky)
    if o.ndim>2:  #3D
        kz = np.fft.fftfreq(o.grid[2],o.meshSize[2])
if o.ndim<2: ky = 0.
if o.ndim<3: kz = 0.

rotR_fx, rotR_fy, rotR_fz = o.crossProduct(kx, ky, kz,
                                           np.fft.fftn(Ex,axes=ax),
                                           np.fft.fftn(Ey,axes=ax),
                                           np.fft.fftn(Ez,axes=ax))

i2pi = 1j * 2*np.pi
rotRx, rotRy, rotRz = (np.fft.ifftn(rotR_fx *i2pi,axes=ax).real,
                       np.fft.ifftn(rotR_fy *i2pi,axes=ax).real,
                       np.fft.ifftn(rotR_fz *i2pi,axes=ax).real)


#%%
#----------------------------------------------
plt.close("all")
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)




