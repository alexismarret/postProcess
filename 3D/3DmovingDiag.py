#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:08:31 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

from matplotlib.patches import Rectangle

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="CS3Drmhr"
o = osiris.Osiris(run,spNorm="iL")

st = slice(None,None,1)
time = o.getTimeAxis()[st]
N = len(time)
dt = time[1]-time[0]

species ="iL"
mass = np.abs(o.rqm[o.sIndex(species)])

#----------------------------------------------
x = o.getAxis("x")
y = o.getAxis("y")
extent=(min(x),max(x),min(y),max(y))
Ncell = o.grid[0]/o.gridPosMax[0]  #number of cells per electron skin depth

#window is fixed along x axis, on depth fixed along z axis, moves along y axis
frac = 1/4
dx = int(Ncell * np.sqrt(mass)*frac)  #number of cells to have fraction frac of wavelength
dy = int(Ncell * np.sqrt(mass)*frac)
dz = int(Ncell * np.sqrt(mass)*frac)

dx*=2
dy*=2
dz*=2

#position along flowing direction (indexes)
start_x = 100
end_x   = start_x+dx
sx = slice(start_x,end_x,1)

#starting depth
start_z = 0
end_z   = start_z+dz
sz = slice(start_z,end_z,1)

#starting height
start_y = 0
end_y   = start_y+dy

sliceFull = (slice(None,None,1),slice(None,None,1),start_z)

#simulate moving diagnostic

#experiment
#c/wpi = 120 micrometer
#wpi = 2.5e12 s^-1 = 2.5e3 ns^-1
#wpi^-1 = 1/2.5e3 ns
#1 ns = 2.5e3 wpi^-1
#time to cross one wavelenght: 0.6 ns = 0.6*2.5e3 wpi^-1
#time between measurements: 0.04 ns = 0.04*2.5e3 wpi^-1, ~12 measurements per wavelength
#growth rate calculated is 5.5 ns^-1

#simulation
#gamma = 0.5 wpi, factor 227.3 times faster growth rate (0.5/0.0022)

#time to cross one wavelength [wpi^-1] = time in experiment mutliplied by growth rate scaling factor
tc = 0.6*2.5e3 / 227.3/4
lambdaFil = 0.5  #wavelength ion filamentation [c/wpi]
#v = 0.5 (c/wpi) / 6.6 (wpi^-1)  [c]
v = lambdaFil  / tc / 2

d = v*dt *dy

#average over diag volume
av = (1,2,3)

#----------------------------------------------
path = o.path+"/plots/ni_movingDiag"
o.setup_dir(path)

n  = np.zeros(N)
Ti = np.zeros(N)
J  = np.zeros(N)
moved = start_y  #keep track of distance travelled

#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

data = o.getCharge(time[0], "iL", sl=sliceFull)

im=sub1.imshow(data.T,
               extent=extent,origin="lower",
               aspect=1,
               cmap="bwr",
               vmin = 0.01, vmax = 1,
               interpolation="None")

divider = make_axes_locatable(sub1)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

sub1.locator_params(nbins=20,axis='x')
sub1.locator_params(nbins=20,axis='y')

sub1.set_xlabel(r'$x\ [c/\omega_{pi}]$')
sub1.set_ylabel(r'$y\ [c/\omega_{pi}]$')

sub1.text(1, 1.05,
          r"$n_i\ [(c^3/\omega_{pe}^3)]$",
          horizontalalignment='right',
          verticalalignment='bottom',
          transform=sub1.transAxes)

txt = sub1.text(0.35, 1.05,
                r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[0],
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=sub1.transAxes)

#needed to avoid change of figsize
plt.savefig(path+"/plot-{i}-time-{t}.png".format(i=0,t=time[0]),dpi="figure")

for i in range(N):

    Artist.remove(txt)

    txt = sub1.text(0.35, 1.05,
                    r"$t=%.1f\ [\omega_{pi}^{-1}]$"%time[i],
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=sub1.transAxes)

    data = o.getCharge(time[i], "iL", sl=sliceFull)
    im.set_array(data.T)

    #draw averaging surface
    rec = sub1.add_patch(Rectangle((x[start_x],y[start_y]),
                                    x[end_x]-x[start_x],
                                    y[end_y]-y[start_y],
                                    edgecolor="cyan",fill=False))

    plt.savefig(path+"/plot-{i}-time-{t}.png".format(i=i,t=time[i]),dpi="figure")
    rec.remove()

    #----------------------------------------------
    sy = slice(start_y,end_y,1)
    sl = (sx,sy,sz)

    n[i] = o.getCharge(time[i], species, sl=sl,av=av)
    Ti[i] = o.getUth(time[i], species, "x", sl=sl,av=av)**2*mass
    J[i] = o.getTotCurrent(time[i], "x", sl=sl,av=av)

    # print(moved,start_y,end_y)
    moved+=d
    start_y = round(moved)
    end_y = round(moved+dy)


#----------------------------------------------
fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,n,color="b",label=r"$<n>$")
sub1.plot(time,Ti,color="g",label=r"$<T_i>$")
# sub1.plot(time,J,color="k",label=r"$<J>$")
sub1.set_xlabel(r"$t\ [\omega_{pi}^{-1}]$")

sub1.legend(frameon=False)
# sub1.set_ylim(0.1,1)

sub1.set_yscale("log")
