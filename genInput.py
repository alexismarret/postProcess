#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:05:45 2022

@author: alexis
"""
import numpy as np
import itertools
import sys

#1D: power of 2**1: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
#2D: power of 2**2: 1, 4, 16, 64, 256, 1024, 4096
#3D: power of 2**3: 1, 8, 64, 512, 4096
#--------------------------------------------------------------
dim = "3D"
Nnodes = 64
NCPUperNodes = 64
Nthreads = 4

Ncell = np.array([512,512,512])
duration = 60              #in units of 1/w_pi

v  = 0.5                    #velocity in units of c (=beta)
n0 = 0.5     #density in proper frame
T  = 1e0    #in units of me * c^2 (=511 KeV) in rest frame

mu = 100.

dx = 1/4      #in units of c/w_pe
dy = 1/4
dz = 1/4

ppc = 8
nPop = 4

dtDump = 0.5    #dump time step desired in units of 1/w_pi

#in units of c/wpi
# zoneX = (0,128)
# zoneY = (0,15)
# zoneZ = (15,25)
# zoneCells = (zoneX[1]-zoneX[0])*(zoneY[1]-zoneY[0])*(zoneZ[1]-zoneZ[0])
# t<400 t>70
#--------------------------------------------------------------
lorentz = 1./np.sqrt(1-v**2)
u = lorentz * v   #momentum
n = lorentz * n0
uthe = np.sqrt(T)
uthi = np.sqrt(T/mu)

# B = 10.
# t_ci_wpe = 1. / (B/(gamma*mu))
ratio_l_i_l_e = np.sqrt(mu)
ratio_l_d_l_e = np.sqrt(T/lorentz)

gammaIfil = v/lorentz  #[wpi]
lambdaIfil = 2*np.pi   #[c/wpi]
R0 = lambdaIfil*np.sqrt(mu)/4  #typical radius of ion filament [c/wpe]

gammaKink = 3/2 * v * np.sqrt(1/(mu*R0))    #[wpi]
lambdaKink = 2*np.pi * 2/3 * np.sqrt(R0)    #[c/wpi]
kKink = 2*np.pi/lambdaKink

#assuming 8 e-foldings and an empyrical factor 5 correction for isotropization time
tEq = (8/gammaIfil + 8/gammaKink)*5

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

    #get multiples of Ncores
    d = np.array([x for x in range(1,Ncores+1) if Ncores % x == 0])
    #get dividors, except product of itself
    c=np.array([x for x in itertools.combinations(d,len(Ncell))
                        if np.product(x)==Ncores])
    #exit if no dividors
    if len(c)==0:
        print("No domain decomposition found for",Ncores,"cores")
        return [0]*len(Ncell), 0

    #check if sqrt or cubic sqrt in dividors
    sq = [p for p in d if p**len(Ncell) == Ncores]
    #add [Ncores**(1/dim)] if integer for 1:1 or 1:1:1 core ratio
    if len(sq)!=0: c = np.vstack((c,sq*len(Ncell)))

    #find ratio of cores closest to 1
    ratio = np.prod(c[:,:-1] / c[:,-1][...,None],axis=1)
    indices = [j for j, v in enumerate(ratio) if v==max(ratio)]

    #assign cores in spatial directions
    coresRep=np.zeros((len(indices),len(Ncell)),dtype=int)
    for j,i in enumerate(indices):

        #argsort(): original indexes of the sorted Ncell array
        #argsort()[-1] is index of largest value of Ncell
        for it,index in enumerate(np.argsort(Ncell)): coresRep[j,index] = c[i,it]

    #measure quality of cells per cores
    std = np.std(Ncell/coresRep,axis=1)

    #maximize cells per core among best choice of ratio
    best = np.where((std)==min(std))[0][0]

    return coresRep[best], ratio[indices[best]]

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
#print("t_ci/t_wpe =",round(t_ci_wpe,r),"| B =",B)
print("mu =",mu)
print("li/le =",round(ratio_l_i_l_e,r))
print("lD/le =",round(ratio_l_d_l_e,r))
print("lorentz =",round(lorentz,r))
print("kmin =",round(kmin,r),
      "| kmax =",round(kmax,r),"[wpi/c] (x)")
print("-------------------------------")

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



