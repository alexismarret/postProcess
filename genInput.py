#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:05:45 2022

@author: alexis
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt
import parallelFunctions as pf

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 2,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 5,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)

#--------------------------------------------------------------
dim = "3D"
Nnodes = 512
NCPUperNodes = 64
Nthreads = 4

Ncell = np.array([512,512,512])
duration = 200              #in units of 1/w_pi

v  = 0.5                    #velocity in units of c (=beta)
n0 = 0.5     #density in proper frame
T  = 1e-6    #in units of me * c^2 (=511 KeV) in rest frame
alfMach = 20   #wanted Alfvenic Mach number
B_angle = 80  #angle between x axis and B in degrees in (x,y) plane
mu = 32

dx = 1/4      #in units of c/w_pe
dy = 1/4
dz = 1/4

ppc = 8
nPop = 4

dtDump = 0.5    #dump time step desired in units of 1/w_pi

scanCoresRep = False
Nmin = 128
Nmax = 256

#--------------------------------------------------------------
lorentz = 1./np.sqrt(1-v**2)
u = lorentz * v   #momentum
n = lorentz * n0
uthe = np.sqrt(T)
uthi = np.sqrt(T/mu)

#B needed to get AlfMach
if alfMach!=0: B  = v * np.sqrt(n0*(1+mu)) / alfMach
else:          B  = 0
#components to satisfy desired angle
Bx = B * np.cos(B_angle*np.pi/180)
By = B * np.sin(B_angle*np.pi/180)

# t_ci_wpe = 1. / (B/(gamma*mu))
ratio_l_i_l_e = np.sqrt(mu)
ratio_l_d_l_e = np.sqrt(T/lorentz)

gammaIfil = v/lorentz  #[wpi]
lambdaIfil = 2*np.pi   #[c/wpi]
R0 = lambdaIfil*np.sqrt(mu)/4  #typical radius of ion filament [c/wpe]

gammaKink = 3/2 * v * np.sqrt(1/(mu*R0))    #[wpi]
lambdaKink = 2*np.pi * 2/3 * np.sqrt(R0)    #[c/wpi]
kKink = 2*np.pi/lambdaKink

#assuming 8 e-foldings and an empyrical factor 3 correction for isotropization time
tEq = (8/gammaIfil + 8/gammaKink)*3

if dim=="1D":
    Ncell = Ncell[0]
    Lx = Ncell*dx
    dt = dx

elif dim=="2D":
    Ncell = Ncell[:2]
    Lx = Ncell[0]*dx
    Ly = Ncell[1]*dy
    dt = 1./np.sqrt((1./dx)**2+(1./dy)**2)

elif dim=="3D":
    Lx = Ncell[0]*dx
    Ly = Ncell[1]*dy
    Lz = Ncell[2]*dz
    dt = 1./np.sqrt((1./dx)**2+(1./dy)**2+(1./dz)**2)

if dim=="1D": dt*=0.9
else:         dt*=0.9999

nIter = int(np.ceil(duration*np.sqrt(mu)/dt))
ndumpTot = int(np.ceil(duration/dtDump))      #total number of dumps desired
nDump = int(np.floor(nIter/ndumpTot))
dtDumpWpe = dtDump*np.sqrt(mu)

nbrPart = ppc*np.product(Ncell)*nPop
t_estimate = 1e-6 * nIter*nbrPart /3600. *3   #1us per part per dt, factor 3 correction
sizeDump = 32./8. / (1024.**3) * np.product(Ncell)*ndumpTot
Ncores = Nnodes*NCPUperNodes
durationWpe = duration*np.sqrt(mu)
num_par_max = int(3*nbrPart/(Nthreads*Ncores))

kmin = 2*np.pi/(Lx/np.sqrt(mu))
kmax = np.pi/(dx/np.sqrt(mu))

#--------------------------------------------------------------
#get domain decomposition
#two constraints: minimize surface to reduce communication needs
#and maximize number of cells per core
def domainDecomposition(Ncores,Ncell):

    Ndim = len(Ncell)
    #get multiples of Ncores
    d = [x for x in range(1,Ncores+1) if Ncores % x == 0]
    #get dividors, except product of itself
    c = np.array([x for x in itertools.combinations(d,Ndim)
                        if np.product(x)==Ncores])
    #exit if no dividors
    if len(c)==0:
        print("No domain decomposition found for",Ncores,"cores")
        return [0]*Ndim, 0

    #check if sqrt or cubic sqrt in dividors
    sq = [p for p in d if p**Ndim == Ncores]
    #add [Ncores**(1/dim)] if integer for 1:1 or 1:1:1 core ratio
    if len(sq)!=0: c = np.vstack((c,sq*Ndim))

    #find ratio of cores closest to 1
    ratio = np.prod(c[:,:-1] / c[:,-1][...,None],axis=1)
    indices = [j for j, v in enumerate(ratio) if v==np.max(ratio)]

    #assign cores in spatial directions
    coresRep = np.zeros((len(indices),Ndim),dtype=int)
    for j,i in enumerate(indices):

        #argsort(): original indexes of the sorted Ncell array
        #argsort()[-1] is index of largest value of Ncell
        for it,index in enumerate(np.argsort(Ncell)): coresRep[j,index] = c[i,it]

    #measure quality of cells per cores
    std = np.std(Ncell/coresRep,axis=1)

    #maximize cells per core among best choice of ratio
    best = np.where(std==min(std))[0][0]

    return coresRep[best], ratio[indices[best]]


#--------------------------------------------------------------
if scanCoresRep:
    # ratioN    = np.zeros(Nmax+1-Nmin)
    # coresRepN = np.zeros((Nmax+1-Nmin,len(Ncell)))
    rge = np.array(range(Nmin,Nmax+1))

    nP = 6
    stages = pf.distrib_task(0, Nmax-Nmin, nP)
    it = ((n*NCPUperNodes, Ncell) for n in rge)

    coresRepN, ratioN = pf.parallel(domainDecomposition, it, nP)
    cond = (ratioN>=0.1)
    fig, (sub1) = plt.subplots(1,figsize=(6,5),dpi=300)

    sub1.axhline(0.1,color="gray",linestyle="--",linewidth=0.7)
    sub1.axhline(1,color="gray",linestyle="--",linewidth=0.7)
    sub1.semilogy(rge[cond],ratioN[cond],linestyle="",marker="x",markersize=2)

    for i, v in enumerate(ratioN[cond]):
        sub1.text(rge[cond][i], v*1.03, "%d" %rge[cond][i], ha="center")

    # sub1.axes.get_xaxis().set_visible(False)
    sub1.set_xlabel(r"$\#\ nodes$")
    sub1.set_ylabel(r"$Aspect\ ratio$")

#--------------------------------------------------------------
if dim!="1D": coresRep, ratio = domainDecomposition(Ncores,Ncell)

#--------------------------------------------------------------
r=5
print("-------------------------------")
if dim=="1D":
    print("Nx =",Ncell[0])
elif   dim=="2D":
    print("Nx =",Ncell[0],"| Ny =",Ncell[1])
elif dim=="3D":
    print("Nx =",Ncell[0],"| Ny =",Ncell[1],"| Nz =",Ncell[2])

print("Lx =",Lx,"[c/wpe] <->",round(Lx/ratio_l_i_l_e,1),"[c/wpi]")
if   dim=="2D":
    print("Ly =",Ly,"[c/wpe] <->",round(Ly/ratio_l_i_l_e,1),"[c/wpi]")
elif dim=="3D":
    print("Ly =",Ly,"[c/wpe] <->",round(Ly/ratio_l_i_l_e,1),"[c/wpi]")
    print("Lz =",Lz,"[c/wpe] <->",round(Lz/ratio_l_i_l_e,1),"[c/wpi]")

print("-------------------------------")
print("tFinal =",round(durationWpe),"[1/wpe] <->",
                 round(duration,1),"[1/wpi]")
print("dt =",round(dt,r))
print("nDump =",nDump)
print("num_par_max =",num_par_max)

print("-------------------------------")
print("n =",round(n,r))
print("u =",round(u,r))
print("uthe =",round(uthe,r))
print("uthi =",round(uthi,r))

print("-------------------------------")
print("v/v_A =",round(alfMach,r),"| B =",round(B,r))
print("B_angle =",str(round(B_angle,r))+"°",
      "| Bx =",round(Bx,r) ,"| By =",round(By,r))
print("mu =",mu)
print("li/le =",round(ratio_l_i_l_e,r))
print("lD/le =",round(ratio_l_d_l_e,r))
print("lorentz =",round(lorentz,r))

print("-------------------------------")
print("kmin =",round(kmin,r),
      "| kmax =",round(kmax,r),"[wpi/c] (x)")
print("gammaIfil =",round(gammaIfil,r),"[wpi]")
print("gammaKink =",round(gammaKink,r),"[wpi]")
print("kKink =",round(kKink,r),"[wpi/c]")
print("tEq =",round(tEq,1),"[1/wpi]")

print("-------------------------------")
print("nIter =",nIter)
print("ndumpTot =",ndumpTot)
print("dtDump =",round(dtDumpWpe,1),"[1/wpe] <->",
                 dtDump,"[1/wpi]")

print("-------------------------------")
print("nbrPart =",round(nbrPart/1e6,2),"millions")
if dim=="1D":
    print("Ncores =",Ncores)
    print("Cells per core =",np.round(Ncell/Ncores,1))
else:
    print("Ncores =",coresRep,"- ratio =",round(ratio,r))
    print("Cells per core =",np.round(Ncell/coresRep,1))

print("-------------------------------")
print("Estimated time:", round(t_estimate/Ncores,1),
      "hours on", Ncores,"CPU cores -",
      round(t_estimate,1),"hours sys")
print("Size of data dump:", round(sizeDump,1),"GB per grid quantity")
print("-------------------------------")



