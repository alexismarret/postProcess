#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:05:45 2022

@author: alexis
"""
import numpy as np


#--------------------------------------------------------------
dim = "2D"
size = np.array([400.,150.,100.])   #in units of c/w_pe
duration = 3000.                     #in units of w_pe
v = 0.1                             #in units of c

T = 0.001    #in units of me * c^2 (=511 KeV)
             #equal to debye over electron inertial length
mu = 100.

B = 10.

#needs to be a fraction
dx = 1/2.
dy = 1/2.
dz = 1/2.

ppc = 36
nPop = 4
ndumpTot = 150

#--------------------------------------------------------------
gamma = 1./np.sqrt(1-v**2)
u = gamma * v
uthe = np.sqrt(T/511.)
uthi = np.sqrt(T/(mu*511.))

t_ci_wpe = 1./(B/mu)

ratio_l_i_l_e = np.sqrt(mu)
ratio_l_d_l_e = np.sqrt(T)

if   dim=="1D":
    Nx = np.int_(size[0]/dx)
    dt = dx
    nbrPart = ppc*Nx*nPop

elif dim=="2D":
    Nx = np.int_(size[0]/dx)
    Ny = np.int_(size[1]/dy)
    dt = 1./np.sqrt((1./dx)**2+(1./dy)**2)
    nbrPart = ppc*Nx*Ny*nPop

elif dim=="3D":
    Nx = np.int_(size[0]/dx)
    Ny = np.int_(size[1]/dy)
    Nz = np.int_(size[2]/dz)
    dt = 1./np.sqrt((1./dx)**2+(1./dy)**2+(1./dz)**2)
    nbrPart = ppc*Nx*Ny*Nz*nPop

if dim=="1D": dt*=0.9
else:         dt*=0.9999

nDump = int(np.ceil((duration/(dt*ndumpTot))))

#--------------------------------------------------------------

print("--------------------------------------------------------------")
print("Nx =",Nx,"| Lx =",size[0])
if   dim=="2D":
    print("Ny =",Ny,"| Ly =",size[1])
elif dim=="3D":
    print("Ny =",Ny,"| Ly =",size[1])
    print("Nz =",Nz,"| Lz =",size[2])

r=5
print("dt =",round(dt,r))

print("u =",round(u,r))
print("uthe =",round(uthe,r))
print("uthi =",round(uthi,r))

print("nDump =",nDump)
print("--------------------------------------------------------------")
print("t_ci/t_wpe =",round(t_ci_wpe,r),"| B =",B)
print("li/le =",round(ratio_l_i_l_e,r))
print("lD/le =",round(ratio_l_d_l_e,r))
print("nbrPart =",nbrPart)

name = ("CS_"+
        "N"+str(Nx)+"x"+str(Ny)+
        "_L"+str(int(size[0]))+"x"+str(int(size[1]))+
        "_Rqm"+str(int(mu))+
        "_v"+str(v).replace(".","")+
        "_Te"+str(T).replace(".","")+
        "_B"+str(int(B))+
        "_ppc"+str(int(ppc)))

print("--------------------------------------------------------------")
print(name)



