#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:05:45 2022

@author: alexis
"""
import numpy as np


#--------------------------------------------------------------
dim = "3D"
Ncores = 1024

Ncell = np.array([1024,512,512])   #in units of c/w_pe
duration = 10000.                     #in units of w_pe
v = 0.1                             #in units of c

T = 1e-6    #in units of me * c^2 (=511 KeV)
             #equal to debye over electron inertial length
mu = 100.

B = 10.

#needs to be a fraction
dx = 1/2.
dy = 1/2.
dz = 1/2.

ppc = 16
nPop = 4
ndumpTot = 600

#--------------------------------------------------------------
gamma = 1./np.sqrt(1-v**2)
u = gamma * v
uthe = np.sqrt(T)
uthi = np.sqrt(T/mu)

t_ci_wpe = 1./(B/mu)

ratio_l_i_l_e = np.sqrt(mu)
ratio_l_d_l_e = np.sqrt(T)

if   dim=="1D":
    Lx = Ncell[0]*dx
    dt = dx
    nbrPart = ppc*Ncell[0]*nPop

elif dim=="2D":
    Lx = Ncell[0]*dx
    Ly = Ncell[1]*dy
    dt = 1./np.sqrt((1./dx)**2+(1./dy)**2)
    nbrPart = ppc*Ncell[0]*Ncell[1]*nPop

elif dim=="3D":
    Lx = Ncell[0]*dx
    Ly = Ncell[1]*dy
    Lz = Ncell[2]*dz
    dt = 1./np.sqrt((1./dx)**2+(1./dy)**2+(1./dz)**2)
    nbrPart = ppc*Ncell[0]*Ncell[1]*Ncell[2]*nPop

if dim=="1D": dt*=0.9
else:         dt*=0.9999

nDump = int(np.ceil((duration/(dt*ndumpTot))))
nIter = int(np.ceil(duration/dt))
t_estimate = 4e-7 * nIter*nbrPart /(3600.*24.)
#--------------------------------------------------------------

print("-------------------------------")
print("Nx =",Ncell[0],"| Lx =",Lx)
if   dim=="2D":
    print("Ny =",Ncell[1],"| Ly =",Ly)
elif dim=="3D":
    print("Ny =",Ncell[1],"| Ly =",Ly)
    print("Nz =",Ncell[2],"| Lz =",Lz)
r=5
print("dt =",round(dt,r))
print("nDump =",nDump)

print("-------------------------------")
print("u =",round(u,r))
print("uthe =",round(uthe,r))
print("uthi =",round(uthi,r))

print("-------------------------------")
print("t_ci/t_wpe =",round(t_ci_wpe,r),"| B =",B)
print("li/le =",round(ratio_l_i_l_e,r))
print("lD/le =",round(ratio_l_d_l_e,r))
print("-------------------------------")

print("nbrPart =",nbrPart)
print("nIter =",nIter)
print("Estimated time:", round(t_estimate/Ncores,1),
      "days on", Ncores,"CPU cores -",
      round(t_estimate*24,1),"hours")
print("-------------------------------")

# name = ("CS_"+
#         "N"+str(Nx)+"x"+str(Ny)+
#         "_L"+str(int(size[0]))+"x"+str(int(size[1]))+
#         "_Rqm"+str(int(mu))+
#         "_v"+str(v).replace(".","")+
#         "_Te"+str(T).replace(".","")+
#         "_B"+str(int(B))+
#         "_ppc"+str(int(ppc)))

# print("--------------------------------------------------------------")
# print(name)



