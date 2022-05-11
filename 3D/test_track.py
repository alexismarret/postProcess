#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:53:41 2022

@author: alexis
"""


#----------------------------------------------
import osiris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import trackParticles as tr
import parallelFunctions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
run  ="CS3D"
o = osiris.Osiris(run,spNorm="iL")
x = o.getAxis("x")
y = o.getAxis("y")
z = o.getAxis("z")


raise ValueError
#----------------------------------------------
track_data = tr.getTrackData(o.path+'/MS/TRACKS/eL-tracks.h5')

#index of macroparticle
index= slice(0,40,1)
sTime = slice(0,50,1)

#plot parameters
show = True
division = 50
pause = 1e-8

#----------------------------------------------
#macroparticle, iteration
t = track_data['t'][index,sTime]
x1 = track_data['x1'][index,sTime]
x2 = track_data['x2'][index,sTime]
x3 = track_data['x3'][index,sTime]
p1 = track_data['p1'][index,sTime]
p2 = track_data['p2'][index,sTime]
p3 = track_data['p3'][index,sTime]
ene = track_data['ene'][index,sTime]
e1 = track_data['E1'][index,sTime]
e2 = track_data['E2'][index,sTime]
e3 = track_data['E3'][index,sTime]
bx = track_data['B1'][index,sTime]
by = track_data['B2'][index,sTime]
bz = track_data['B3'][index,sTime]

# d_ene_dt = np.gradient(ene)
lorentz = np.sqrt(1+p1**2+p2**2+p3**2)
normb2 = bx**2+by**2+bz**2
sIn=o.sIndex("eL")
pCharge = -1/(o.ppc[sIn]*o.n0[sIn])

#velocity projection in magnetic field aligned basis
v_para  = o.projectVec(p1/lorentz,
                       p2/lorentz,
                       p3/lorentz,
                       bx,by,bz, comp=0)

v_perp1 = o.projectVec(p1/lorentz,
                       p2/lorentz,
                       p3/lorentz,
                       bx,by,bz, comp=1)

v_perp2 = o.projectVec(p1/lorentz,
                       p2/lorentz,
                       p3/lorentz,
                       bx,by,bz, comp=2)

#----------------------------------------------
time = o.getTimeAxis()[sTime]
curv_vX = np.zeros(t.shape)
curv_vY = np.zeros(t.shape)
curv_vZ = np.zeros(t.shape)

#loop over field time:
for i in range(len(time)):

    #get curvature vector for all macroparticles at time i
    kappaX, kappaY, kappaZ = o.magCurv(x1[:,i], x2[:,i], x3[:,i],
                                       bx[:,i], by[:,i], bz[:,i], time[i])

    #calculate curvature drift
    curv_vX[:,i], curv_vY[:,i], curv_vZ[:,i] = (lorentz[:,i]*v_para[:,i]**2/normb2[:,i] *
                                                o.crossProduct(bx[:,i], by[:,i], bz[:,i],
                                                               kappaX, kappaY, kappaZ))

#----------------------------------------------
n1 = np.sqrt((p1/lorentz)**2+(p2/lorentz)**2+(p3/lorentz)**2)
n2 = np.sqrt(v_para**2+v_perp1**2+v_perp2**2)
n3 = np.sqrt(curv_vX**2+curv_vY**2+curv_vZ**2)

work1 = pCharge * e1*p1 / lorentz
work2 = pCharge * e2*p2 / lorentz
work3 = pCharge * e3*p3 / lorentz
# work = work1+work2+work3

#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

#loop over macroparticles
for p in range(len(t)):
    sub1.plot(t[p],n1[p],color="b")
    sub1.plot(t[p],n2[p],color="r")
    sub1.plot(t[p],n3[p],color="g")


raise ValueError

"""
#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
sub1.plot(t,d_ene_dt,color="r")
sub1.plot(t,work,color="b")
"""

#----------------------------------------------
if not show:
    plt.switch_backend('Agg')
    o.setup_dir(o.path+"/plots/track")
#----------------------------------------------
fig = plt.figure(figsize=(4.1,2.8),dpi=200)

gs = GridSpec(3,1, figure=fig)

# sub1 = fig.add_subplot(gs[0,:],projection='3d')

sub1 = fig.add_subplot(gs[0,:])
sub2 = fig.add_subplot(gs[1,:])
sub3 = fig.add_subplot(gs[2,:])

# sub1.set_xlim(min(x),max(x)+x[1]-x[0])
# sub1.set_ylim(min(y),max(y)+y[1]-y[0])
# sub1.set_zlim(min(z),max(z)+z[1]-z[0])

# sub1.set_xlim(min(x),x[2]-x[0])
# sub1.set_ylim(min(y),y[2]-y[0])
# sub1.set_zlim(min(z),z[2]-z[0])

# sub1.set_xlim(min(x1),max(x1))
# sub1.set_ylim(min(x2),max(x2))
# sub1.set_zlim(min(x3),max(x3))

sub2.set_xlim(min(t),max(t))
sub2.set_ylim(min(ene),max(ene))

sub3.set_xlim(min(t),max(t))
sub3.set_ylim(min((min(work1),min(work2),min(work3))),
              max((max(work1),max(work2),max(work3))))

# sub1.set_xlabel(r'$x\ [c/\omega_{pe}]$')
# sub1.set_ylabel(r'$y\ [c/\omega_{pe}]$')
# sub1.set_zlabel(r'$z\ [c/\omega_{pe}]$')

sub2.set_ylabel(r'$\mathcal{E}_{kin}\ [m_ec^2]$')

sub3.set_xlabel(r'$t\ [\omega_{pi}^{-1}]$')
sub3.set_ylabel(r'$q\bf E\cdot\bf v$')

fig.subplots_adjust(top=0.903,
                    bottom=0.194,
                    left=0.300,
                    right=0.963,
                    hspace=0.634,
                    wspace=0.191)

#----------------------------------------------
sub2.plot(t, ene,color="k")

sub3.plot(t, work1,color="r")
sub3.plot(t, work2,color="g")
sub3.plot(t, work3,color="b")



"""
#----------------------------------------------
steps = pf.distrib_task(0, len(t)-1, division)

start=True
for s in steps:

    if start and not show:
        plt.savefig(o.path+"/plots/track"+"/plot-{i}-time-{t}.png".format(i=s[0],t=t[s[0]]),dpi="figure")
        start=False

    sl=slice(s[0],s[1]+1,1)

    # sub1.plot(x1[sl], x2[sl], x3[sl],color="k")

    sub2.plot(t[sl], ene[sl],color="k")

    sub3.plot(t[sl], work1[sl],color="r")
    sub3.plot(t[sl], work2[sl],color="g")
    sub3.plot(t[sl], work3[sl],color="b")

    fig.subplots_adjust(top=0.903,
                        bottom=0.194,
                        left=0.300,
                        right=0.963,
                        hspace=0.634,
                        wspace=0.191)
    if show:
        plt.pause(pause)
    else:
        plt.savefig(o.path+"/plots/track"+"/plot-{i}-time-{t}.png".format(i=s[0],t=t[s[0]]),dpi="figure")

if not show: plt.switch_backend('Qt5Agg')
"""




