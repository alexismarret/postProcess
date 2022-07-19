#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:53:41 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import trackParticles as tr

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 0.9,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
# run  ="CS3Dtrack"
run ="CS3Dtrack"
spNorm = "iL"
o = osiris.Osiris(run,spNorm=spNorm)

species ="iL"

#----------------------------------------------
#index of macroparticle
sPart = slice(None,None,1)
sTime = slice(None,None,1)
sl = (sPart,sTime)

curveDrift = False
maxwellian = False
checkEnergy = True
mult = True

#plot parameters
show = True
dispMax=True
dispMean=False
# division = 1
pause = 1e-8
alpha = 2e-2

#----------------------------------------------
muNorm  = o.rqm[o.sIndex(spNorm)]
mu      = o.rqm[o.sIndex(species)]
pCharge = mu / np.abs(mu)

#macroparticle, iteration
ene = o.getTrackData(species, "ene",sl=sl) * np.abs(mu)
t   = o.getTrackData(species, "t",  sl=sl)
p1  = o.getTrackData(species, "p1", sl=sl)
p2  = o.getTrackData(species, "p2", sl=sl)
p3  = o.getTrackData(species, "p3", sl=sl)
e1  = o.getTrackData(species, "E1", sl=sl)
e2  = o.getTrackData(species, "E2", sl=sl)
e3  = o.getTrackData(species, "E3", sl=sl)





imax = np.where(ene[:,-1]==np.max(ene[:,-1]))[0][0]  #index of most energetic particle

lorentz = np.sqrt(1+p1**2+p2**2+p3**2)
p1 /= lorentz
p2 /= lorentz
p3 /= lorentz

work1 = pCharge * e1*p1
work2 = pCharge * e2*p2
work3 = pCharge * e3*p3
work = work1+work2+work3

intWork1 = cumulative_trapezoid(work1,t,axis=-1,initial=0)
intWork2 = cumulative_trapezoid(work2,t,axis=-1,initial=0)
intWork3 = cumulative_trapezoid(work3,t,axis=-1,initial=0)
intWork  = intWork1+intWork2+intWork3

# t /= np.sqrt(np.abs(muNorm))
#----------------------------------------------
# if not show:
#     plt.switch_backend('Agg')
#     o.setup_dir(o.path+"/plots/track")

if checkEnergy:
    #----------------------------------------------
    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

    dt = (t[0,1]-t[0,0])
    d_ene_dt = np.gradient(ene[imax],dt)
    # d_ene_dt2 = (ene[1:]-ene[:-1])/dt

    sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
    sub1.plot(t[imax]-dt/2,d_ene_dt,color="r",label=r"$d_t\ ene$")
    # sub1.plot(t[1:],d_ene_dt2,color="orange",label=r"$d_t\ ene$",linestyle="--")
    sub1.plot(t[imax],work[imax],color="b",label=r"$W$")

    sub1.legend(frameon=False)


if mult:
    #----------------------------------------------
    fig = plt.figure(figsize=(4.1,2.8),dpi=300)

    # gs = GridSpec(3,1, figure=fig)
    gs = GridSpec(1,1, figure=fig)

    # sub1 = fig.add_subplot(gs[0,:],projection='3d')

    # sub1 = fig.add_subplot(gs[0,:])
    # sub2 = fig.add_subplot(gs[1,:])
    # sub3 = fig.add_subplot(gs[2,:])

    sub1 = fig.add_subplot(gs[0,0])

    # sub1.set_xlim(min(x),max(x)+x[1]-x[0])
    # sub1.set_ylim(min(y),max(y)+y[1]-y[0])
    # sub1.set_zlim(min(z),max(z)+z[1]-z[0])

    sub1.set_xlim(np.min(t),np.max(t))
    sub1.set_ylim(-1.5,np.max(ene))

    sub1.set_ylabel(r'$\mathcal{E}\ [m_ec^2]$')
    sub1.set_xlabel(r'$t\ [\omega_{pi}^{-1}]$')


    fig.subplots_adjust(top=0.903,
                        bottom=0.194,
                        left=0.300,
                        right=0.963,
                        hspace=0.634,
                        wspace=0.191)

    sub1.axhline(0,color="gray",linestyle="--",linewidth=0.7)
    for i in range(len(t)):
        sub1.plot(t[i], ene[i],color="k",alpha=alpha)

        # sub1.plot(t[i], intWork[i],color="orange")

        # sub2.plot(t[i], intWork1[i],color="r")
        # sub3.plot(t[i], intWork2[i],color="g")

    #----------------------------------------------
    if dispMax:
        sub1.plot(t[imax],ene[imax],color="r",
                  label=r"$\max(\mathcal{E}_{kin})$")

        sub1.plot(t[imax],intWork1[imax],color="g",linestyle="-",
                  label=r"$\max(\mathrm{q}\int\bf{v}\cdot\bf{E}_x \mathrm{dt})$")
        sub1.plot(t[imax],intWork2[imax],color="b",linestyle="-",
                  label=r"$\max(\mathrm{q}\int\bf{v}\cdot\bf{E}_y \mathrm{dt})$")
        sub1.plot(t[imax],intWork3[imax],color="orange",linestyle="-",
                  label=r"$\max(\mathrm{q}\int\bf{v}\cdot\bf{E}_z \mathrm{dt})$")
        # sub1.plot(t[imax],intWork[imax],color="b",linestyle="-",
        #           label=r"$\max(\mathrm{q}\int\bf{v}\cdot\bf{E} \mathrm{dt})$")

        sub1.plot(t[imax],intWork[imax]+ene[imax,0],color="k",linestyle="dotted",
                  label=r"$\max(\mathrm{q}\int\bf{v}\cdot\bf{E} \mathrm{dt})+\mathcal{E}_0$")

    #----------------------------------------------
    if dispMean:
        sub1.plot(t[0],np.mean(ene,axis=0),color="r",
                  label=r"$\langle\mathcal{E}_{kin}\rangle$")

        sub1.plot(t[0],np.mean(intWork1,axis=0),color="g",linestyle="-",
                  label=r"$\langle \mathrm{q}\int\bf{v}\cdot\bf{E}_x \mathrm{dt}\rangle$")
        sub1.plot(t[0],np.mean(intWork2,axis=0),color="b",linestyle="-",
                  label=r"$\langle \mathrm{q}\int\bf{v}\cdot\bf{E}_y \mathrm{dt}\rangle$")
        sub1.plot(t[0],np.mean(intWork3,axis=0),color="orange",linestyle="-",
                  label=r"$\langle \mathrm{q}\int\bf{v}\cdot\bf{E}_z \mathrm{dt}\rangle$")


    sub1.legend(frameon=False,loc="upper right")

#----------------------------------------------
if maxwellian:
    def fitEnergyMaxwellian(X, Y, pinit=[1,1]):
        from scipy.optimize import leastsq

        fitfunc=lambda p,x : p[0]/p[1]**(3/2) *2*np.sqrt(x/np.pi) * np.exp(-x/p[1])
        errfunc=lambda p,x,y : y-fitfunc(p,x)

        out=leastsq(errfunc,pinit,args=(X,Y),full_output=1)

        pfinal=out[0]

        n=pfinal[0]
        T=pfinal[1]

        return n, T


    #----------------------------------------------
    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

    sub1.set_xlim(0,5)
    sub1.set_ylim(0,70)
    for i in range(len(t[0])):

        histogram, bins = np.histogram(ene[:,i],bins=200)
        bins = bins[:-1]

        n,T = fitEnergyMaxwellian(bins, histogram, pinit=[0.5,0.1])
        fit = n/T**(3/2) *2*np.sqrt(bins/np.pi) * np.exp(-bins/T)



        l1 = sub1.plot(bins,histogram,color="r")
        l2 = sub1.plot(bins,fit,color="orange")

        if i==0: sub1.legend(loc='upper left')

        sub1.set_title(r"$t={ti}\ [\Omega_0^{{-1}}]$".format(ti=t[0,i]))

        plt.pause(0.1)
        plt.savefig(o.path+"/plots/track"+ "/plot-{i}-time-{t}.png".format(i=i,t=t[0,i]),dpi="figure")
        sub1.lines.remove(l1[0])
        sub1.lines.remove(l2[0])


#----------------------------------------------
if curveDrift:

    x1  = tr.getTrackData(o.path,species,"x1" )[sl]
    x2  = tr.getTrackData(o.path,species,"x2" )[sl]
    x3  = tr.getTrackData(o.path,species,"x3" )[sl]

    bx  = tr.getTrackData(o.path,species,"B1" )[sl]
    by  = tr.getTrackData(o.path,species,"B2" )[sl]
    bz  = tr.getTrackData(o.path,species,"B3" )[sl]

    normb2 = bx**2+by**2+bz**2

    #velocity projection in magnetic field aligned basis
    v_para  = o.projectVec(p1,p2,p3,
                           bx,by,bz, comp=0)

    v_perp1 = o.projectVec(p1,p2,p3,
                           bx,by,bz, comp=1)

    v_perp2 = o.projectVec(p1,p2,p3,
                           bx,by,bz, comp=2)

    #----------------------------------------------
    time = o.getTimeAxis()[sTime]
    curv_vX = np.zeros(t.shape)
    curv_vY = np.zeros(t.shape)
    curv_vZ = np.zeros(t.shape)

    #loop over field time:
    for i in range(len(time)):

        pos = (x1[:,i], x2[:,i], x3[:, i])
        kappaX, kappaY, kappaZ = o.magCurv(pos, bx[:,i], by[:,i], bz[:,i], time[i])

        #calculate curvature drift
        curv_vX[:,i], curv_vY[:,i], curv_vZ[:,i] = (lorentz[:,i]*v_para[:,i]**2/normb2[:,i] *
                                                    o.crossProduct(bx[:,i], by[:,i], bz[:,i],
                                                                   kappaX, kappaY, kappaZ))

    #----------------------------------------------
    n1 = np.sqrt((p1)**2+(p2)**2+(p3)**2)
    n2 = np.sqrt(v_para**2+v_perp1**2+v_perp2**2)
    n3 = np.sqrt(curv_vX**2+curv_vY**2+curv_vZ**2)

    #----------------------------------------------
    fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300,sharex=True,sharey=True)

    #loop over macroparticles
    for p in range(len(t)):
        sub1.plot(t[p],n1[p],color="b")
        sub1.plot(t[p],n2[p],color="r")
        sub1.plot(t[p],n3[p],color="g")








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




