#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:10:26 2022

@author: alexis
"""

import glob
import os

import numpy as np
import operator

import h5py
import trackParticles as tr
import parallelFunctions as pf

import matplotlib.pyplot as plt
import time as ti



class Osiris:

    #--------------------------------------------------------------
    def __init__(self, run, spNorm=None, nbrCores=6):

        self.path = os.environ.get("OSIRIS_RUN_DIR") + "/" + run
        self.allRuns = np.sort(os.listdir(os.environ.get("OSIRIS_RUN_DIR")))
        self.nbrCores = nbrCores

        self.parseInput(run)

        try:    self.normFactor = np.sqrt(np.abs(self.rqm[self.sIndex(spNorm)]))
        except: self.normFactor = 1.

        self.ndim = len(self.grid)
        self.boxSize = self.gridPosMax - self.gridPosMin
        self.meshSize = self.boxSize / self.grid
        self.cellHyperVolume = np.prod(self.meshSize)
        self.boxHyperVolume  = np.prod(self.boxSize)

        return


    #--------------------------------------------------------------
    def parseInput(self, run):

        #open input file
        try:
            input_file = glob.glob(self.path+"/*.in")[0]
        except IndexError:
            raise ValueError("Cannot find input file "+run+".in"+" in '"+self.path+"'")

        with open(input_file) as f: inputs = f.readlines()

        Ns=0
        s=0
        cat=""

        for l in inputs:

            #remove brackets, spaces, line breaks
            l = l.replace("{","").replace("}","").replace(" ","").replace("\n","")

            #skip empty and commented lines
            if l=="" or l[0]=="!": continue

            #find needed categories
            if ((l=="diag_emf")     or
                (l=="diag_species") or
                (l=="diag_current")): cat = l

            #filter for numerical inputs
            elif "=" in l:
                #remove d0 notation and quote
                l = l.replace("d0","").replace('"','')

                #handle comments next to value, remove last comma
                try:               value = l[l.index("=")+1:l.index("!")-1]
                except ValueError: value = l[l.index("=")+1:-1]

                #---------------------------
                #grid parameters
                if "nx_p" in l:
                    self.grid = np.int_(value.split(","))

                elif "xmin" in l:
                    self.gridPosMin = np.float_(value.split(","))

                elif "xmax" in l:
                    self.gridPosMax = np.float_(value.split(","))

                #---------------------------
                #time parameters
                elif "dt=" in l:
                    self.dt = float(value)

                elif "ndump=" in l:
                    self.ndump = int(value)

                elif "tmin=" in l:
                    self.tmin = float(value)

                elif "tmax=" in l:
                    self.tmax = float(value)

                #---------------------------
                #EM parameters
                elif "ext_b0" in l:
                    self.ext_b0 = np.float_(value.split(","))

                elif "ndump_fac_ene_int=" in l:
                    self.ndump_fac_ene_int = int(value)

                #---------------------------
                #particles parameters
                elif ("num_species=" in l) or ("num_cathode=" in l):
                    Ns+=int(value)
                    self.ndump_facP       = np.zeros(Ns)
                    self.ndump_fac_ene    = np.zeros(Ns)
                    self.ndump_fac_raw    = np.zeros(Ns)
                    self.ndump_fac_tracks = np.zeros(Ns)
                    self.niter_tracks     = np.zeros(Ns)
                    self.species_name     = np.empty(Ns, dtype='object')
                    self.rqm              = np.zeros(Ns)
                    self.ppc              = np.zeros(Ns)
                    self.n0               = np.zeros(Ns)
                    self.ufl              = np.zeros((Ns,3))
                    self.uth              = np.zeros((Ns,3))

                elif "name=" in l:
                    if self.species_name[s] != None: s+=1
                    self.species_name[s] = value

                elif "rqm=" in l:
                    if self.rqm[s] != 0: s+=1
                    self.rqm[s] = float(value)

                elif "num_par_x" in l:
                    if self.ppc[s] != 0: s+=1
                    self.ppc[s] = np.prod(np.int_(value.split(",")))

                elif "density=" in l:
                    self.n0[s] = float(value)

                elif "ufl(" in l:
                    self.ufl[s] = np.float_(value.split(","))

                elif "uth(" in l:
                    self.uth[s] = np.float_(value.split(","))

                #---------------------------
                #ndump parameters
                elif cat=="diag_species":
                    if "ndump_fac=" in l:
                        self.ndump_facP[s] = int(value)

                    elif "ndump_fac_ene=" in l:
                        self.ndump_fac_ene[s] = int(value)

                    elif "ndump_fac_raw=" in l:
                        self.ndump_fac_raw[s] = int(value)

                    elif "ndump_fac_tracks=" in l:
                        self.ndump_fac_tracks[s] = int(value)

                    elif "niter_tracks=" in l:
                        self.niter_tracks[s] = int(value)

                elif cat=="diag_current":
                    if "ndump_fac=" in l:
                        self.ndump_facC = int(value)

                elif cat=="diag_emf":
                    if "ndump_fac=" in l:
                        self.ndump_facF = int(value)

        return


    #--------------------------------------------------------------
    def printAttributes(self):

        print("-------------------------------")
        print("path =", self.path)
        print("ndim =", self.ndim)
        print("grid =", self.grid)
        print("gridPosMin =", self.gridPosMin,"|","gridPosMax =", self.gridPosMax)
        print("boxSize =",self.boxSize)
        print("meshSize =",self.meshSize)
        print("cellHyperVolume =", self.cellHyperVolume)
        print("boxHyperVolume =", self.boxHyperVolume)

        print("-------------------------------")
        print("dt =", self.dt)
        print("tmin =", self.tmin)
        print("tmax =", self.tmax)
        try: print("ext_b0 =", self.ext_b0)
        except: pass

        print("-------------------------------")
        print("ndump =", self.ndump)
        try: print("ndump_fac_ene_int =", self.ndump_fac_ene_int)
        except: pass
        try: print("ndump_facF =", self.ndump_facF)
        except: pass
        try: print("ndump_facC =", self.ndump_facC)
        except: pass

        print("-------------------------------")
        print("species_name =", self.species_name)
        print("rqm =", self.rqm)
        print("ppc =", self.ppc)
        print("n0 =", self.n0)
        print("ufl =", self.ufl)
        print("uth =", self.uth)

        print("-------------------------------")
        print("ndump_facP =", self.ndump_facP)
        print("ndump_fac_ene =", self.ndump_fac_ene)
        print("ndump_fac_raw =", self.ndump_fac_raw)
        print("ndump_fac_tracks =", self.ndump_fac_tracks)
        print("niter_tracks =", self.niter_tracks)

        return


    #--------------------------------------------------------------
    def sIndex(self, species):

        try:
            index = np.where(self.species_name==species)[0][0]
        except:
            raise ValueError("Unknown species '"+species+"'")

        return index


    #--------------------------------------------------------------
    def getAxis(self, direction):

        if   direction == "x": i=0
        elif direction == "y": i=1
        elif direction == "z": i=2

        axis = np.linspace(self.gridPosMin[i],(self.grid[i]-1)*self.meshSize[i],self.grid[i])

        return axis / self.normFactor


    #--------------------------------------------------------------
    def getTimeAxis(self, species=None, ene=False, raw=False, track=False):

        #species time
        if species!=None:
            species_index = self.sIndex(species)

            if ene:
                sIndex = species_index + 1
                if sIndex < 10: sIndex = "0" + str(sIndex)   #make sure padding is correct
                N = len(np.loadtxt(self.path+"/HIST/par"+str(sIndex)+"_ene",skiprows=2,usecols=2))
                multFactor = self.ndump_fac_ene[species_index]

            elif raw:
                #use glob.glob to ignore hidden files
                N = len(glob.glob(self.path+"/MS/RAW/"+species+"/*"))
                multFactor = self.ndump_fac_raw[species_index]

            else:
                #retrieve number of dumps from any of the folders in /DENSITY
                N = len(glob.glob(self.path+"/MS/UDIST/"+species+"/"+
                                   os.listdir(self.path+"/MS/UDIST/"+species)[0]+"/*"))
                multFactor = self.ndump_facP[species_index]

        #fields time
        else:
            if ene:
                N = len(np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=2))
                multFactor = self.ndump_fac_ene_int

            else:
                #retrieve number of dumps from any of the folders in /FLD
                N = len(glob.glob(self.path+"/MS/FLD/"+
                                   os.listdir(self.path+"/MS/FLD")[0]+"/*"))
                multFactor = self.ndump_facF

        delta = self.dt*self.ndump*multFactor
        time = np.linspace(self.tmin,(N-1)*delta,N) / self.normFactor

        return time.round(7)



    #--------------------------------------------------------------
    def getOnGrid(self, time, dataPath, species, sl, av, parallel, transpose):

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        #get field or species times
        index=np.nonzero(np.in1d(self.getTimeAxis(species),time))[0]

        #check if requested times exist
        if len(index)!=N: raise ValueError("Unknown time for '"+dataPath+"'")

        #complete slice if needed
        slices = [slice(None)]*self.ndim
        if type(sl) in {slice,int}: sl = (sl,)
        for k,s in enumerate(sl): slices[k]=s

        #invert slices because of needed transposition
        #slices performance can be worse than reading everything
        #does not support list of unordered indices
        slices = tuple(slices)[::-1]

        #adjust axis average, reversed because of transposition
        #av input can be 1,2,3 corresponding to spatial axis only
        if av!=None:
            if type(av)==int: av = (av,)
            av = self.revertAx((a-1 for a in av))

        #create inputs
        it = ((dataPath + p, slices, av, transpose) for p in np.take(sorted(os.listdir(dataPath)), index))

        #multiple values read
        if N>1:
            #parallel reading of data
            if parallel:
                G = pf.parallel(pf.readGridData, it, self.nbrCores)
            #sequential reading of data
            else:
                #calculate size of sliced array, invert again slices and averaged
                #axis to order after transposition
                G = np.zeros((N,)+self.getSlicedSize(slices[::-1],self.revertAx(av)))
                for i in range(N):
                    G[i] = pf.readGridData(next(it)[0], slices, av, transpose)

        #single value read
        else:
            G = pf.readGridData(next(it)[0], slices, av, transpose)

        return G


    #--------------------------------------------------------------
    def revertAx(self, a):

        if a==None:
            return (None,)
        else:
            val = self.ndim-1
            a = list(a)
            for i in range(len(a)):
                if   a[i] == 0: a[i] = val
                elif a[i] == val: a[i] = 0

            return tuple(a)


    #--------------------------------------------------------------
    def getSlicedSize(self, sl, av):

        #array containing final indices and step of sliced array
        #None if no slicing in a given direction, and 0 if only one element
        ind = [None]*self.ndim

        for k in range(len(ind)):
            #if axis is averaged or single element, set size to 0 and go to next axis
            if (k in av) or (type(sl[k])==int): ind[k] = 0
            #element of sl is a slice()
            elif type(sl[k])==slice: ind[k] = sl[k].indices(self.grid[k])

        #get corresponding number of elements accounting for uneven divisions
        sh = []
        for k,i in enumerate(ind):
            if type(i)==tuple:
                if (i[1]-i[0])%i[2]!=0: sh.append((i[1]-i[0])//i[2]+1)
                else:                   sh.append((i[1]-i[0])//i[2])
            elif i==None:
                sh.append(self.grid[k])

        return tuple(sh)


    #--------------------------------------------------------------
    def getEnergyIntegr(self, time, qty, species=None):

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        cond=np.in1d(self.getTimeAxis(species=species,ene=True),time)

        #check if requested times exist
        if len(np.nonzero(cond)[0])!=N: raise ValueError("Unknown time for '"+qty+"'")

        #energy per field component
        if qty=="B":
            ene_Bx = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=2)[cond] / self.boxHyperVolume
            ene_By = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=3)[cond] / self.boxHyperVolume
            ene_Bz = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=4)[cond] / self.boxHyperVolume

            return ene_Bx, ene_By, ene_Bz

        elif qty=="E":
            ene_Ex = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=5)[cond] / self.boxHyperVolume
            ene_Ey = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=6)[cond] / self.boxHyperVolume
            ene_Ez = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=7)[cond] / self.boxHyperVolume

            return ene_Ex, ene_Ey, ene_Ez

        #kinetic energy
        elif qty=="kin":
            sIndex = self.sIndex(species) + 1
            #make sure padding is correct
            if sIndex < 10: sIndex = "0" + str(sIndex)
            else:           sIndex = str(sIndex)

            ene = np.loadtxt(self.path+"/HIST/par"+sIndex+"_ene",skiprows=2,usecols=3)[cond] / self.boxHyperVolume

            return ene



    #--------------------------------------------------------------
    def getB(self, time, comp, sl=slice(None), av=None,
             parallel=True, transpose=True):

        if   comp=="x": key = "b1"
        elif comp=="y": key = "b2"
        elif comp=="z": key = "b3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        B = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return B


    #--------------------------------------------------------------
    def getE(self, time, comp, sl=slice(None), av=None,
             parallel=True, transpose=True):

        if   comp=="x": key = "e1"
        elif comp=="y": key = "e2"
        elif comp=="z": key = "e3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        E = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return E


    #--------------------------------------------------------------
    def getUfluid(self, time, species, comp, sl=slice(None), av=None,
                  parallel=True, transpose=True):

        if   comp=="x": key = "ufl1"
        elif comp=="y": key = "ufl2"
        elif comp=="z": key = "ufl3"

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Ufluid = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Ufluid


    #--------------------------------------------------------------
    def getUth(self, time, species, comp, sl=slice(None), av=None,
               parallel=True, transpose=True):

        if   comp=="x": key = "uth1"
        elif comp=="y": key = "uth2"
        elif comp=="z": key = "uth3"

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Uth = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Uth


    #--------------------------------------------------------------
    def getTemp(self, time, species, comp, sl=slice(None), av=None,
                parallel=True, transpose=True):

        key = "T"+comp

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        T = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return T


    #--------------------------------------------------------------
    def getCharge(self, time, species, cellAv=False, sl=slice(None), av=None,
                  parallel=True, transpose=True):

        """
        Get species charge density C = n*q
        cellAv: moment obtained from macroparticles in the cell, no interpolation
        """

        key = "charge"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Chr = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Chr


    #--------------------------------------------------------------
    def getMass(self, time, species, cellAv=False, sl=slice(None),
                      av=None, parallel=True, transpose=True):

        """
        Get species mass density M = n*m
        """

        key = "m"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        M = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return M


    #--------------------------------------------------------------
    def getCurrent(self, time, species, comp, cellAv=False, sl=slice(None), av=None,
                   parallel=True, transpose=True):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Cur = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Cur


    #--------------------------------------------------------------
    def getTotCurrent(self, time, comp, sl=slice(None), av=None,
                      parallel=True, transpose=True):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        totCur = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return totCur


    #--------------------------------------------------------------
    #get species kinetic energy density
    def getKinEnergy(self, time, species, cellAv=False, sl=slice(None), av=None,
                     parallel=True, transpose=True):

        key = "ene"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Enrgy = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Enrgy


    #--------------------------------------------------------------
    def getRaw(self, time, species, key, parallel=True):

        #['SIMULATION', 'ene', 'p1', 'p2', 'p3', 'q', 'tag', 'x1', 'x2', 'x3']

        dataPath = self.path+"/MS/RAW/"+species+"/"

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        #get time
        index=np.nonzero(np.in1d(self.getTimeAxis(species,raw=True),time))[0]

        #check if requested times exist
        if len(index)!=N: raise ValueError("Unknown time for '"+dataPath+"'")

        #create inputs
        it = ((dataPath + p, key) for p in np.take(sorted(os.listdir(dataPath)), index))

        #multiple values read, very heavy in memory because of irregular data shape
        if N>1:
            #parallel reading of data
            if parallel:
                G = pf.parallel(pf.readRawData, it, self.nbrCores)
            #sequential reading of data
            else:
                G = np.asarray(tuple(pf.readRawData(i[0], key) for i in it), dtype=object)

        #single value read
        else:
            G = pf.readRawData(next(it)[0], key)

        return G


    #--------------------------------------------------------------
    def getPhaseSpace(self, time, species, direction, comp,
                      parallel=True, transpose=True,):

        if    direction=="x": l = "x1"
        elif  direction=="y": l = "x2"
        elif  direction=="z": l = "x3"

        if   comp=="x": p = "p1"
        elif comp=="y": p = "p2"
        elif comp=="z": p = "p3"
        elif comp=="g": p = "gamma"

        dataPath = self.path+"/MS/PHA/"+p+l+"/"+species+"/"

        Pha = self.getOnGrid(time,dataPath,species,parallel)

        return Pha


    #--------------------------------------------------------------
    def getNewData(self, time, key, sl=slice(None), av=None,
                   parallel=True, transpose=False):

        dataPath = self.path+"/PP/"+key+"/"

        #data needs to be transposed back
        D = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return D


    #--------------------------------------------------------------
    def setup_dir(self, path, rm = True):

        if os.path.exists(path) and rm:
            for ext in ("png","eps","pdf","mp4"):
                for file in glob.glob(path+"/*."+ext): os.remove(file)

        elif not os.path.exists(path):
            os.makedirs(path)

        return


    #--------------------------------------------------------------
    def crossProduct(self, Ax, Ay, Az, Bx, By, Bz):

        cx = Ay*Bz - Az*By
        cy = Az*Bx - Ax*Bz
        cz = Ax*By - Ay*Bx

        return cx, cy, cz


    #--------------------------------------------------------------
    def projectVec(self, vx, vy, vz, ex, ey, ez, comp):

        #assumes e!= (0,1,0)
        Ix, Iy, Iz = 0., 1., 0.

        #project vector 'v' in basis 'e' along direction 'comp'
        if comp==0:   #parallel to 'e'
            baseX, baseY, baseZ = ex, ey, ez

        else: #e x unit vector I
            baseX, baseY, baseZ = self.crossProduct(ex, ey, ez, Ix, Iy, Iz)

            if comp==2: #e x (e x unit vector I)
                baseX, baseY, baseZ = self.crossProduct(ex, ey, ez, baseX, baseY, baseZ)

        return (vx*baseX + vy*baseY + vz*baseZ) / np.sqrt(baseX**2+baseY**2+baseZ**2)


    #--------------------------------------------------------------
    def findCell(self, pos):

        #finds cell indexes from macroparticle positions
        if self.ndim==1:

            i = np.int_(pos // self.meshSize[0])
            #need to check that index different than number of cells
            i[i==self.grid[0]] = self.grid[0]-1

            return i

        elif self.ndim==2:

            i = np.int_(pos[0] // self.meshSize[0])
            j = np.int_(pos[1] // self.meshSize[1])

            i[i==self.grid[0]] = self.grid[0]-1
            j[j==self.grid[1]] = self.grid[1]-1

            return i, j

        elif self.ndim==3:

            i = np.int_(pos[0] // self.meshSize[0])
            j = np.int_(pos[1] // self.meshSize[1])
            k = np.int_(pos[2] // self.meshSize[2])

            i[i==self.grid[0]] = self.grid[0]-1
            j[j==self.grid[1]] = self.grid[1]-1
            k[k==self.grid[2]] = self.grid[2]-1

            return i, j, k


    #--------------------------------------------------------------
    def magCurv(self, pos, bx, by, bz, time):

        #calculate magnetic field curvature at macroparticle position
        #need position as pos = x1; tuple(x1,x2); tuple(x1,x2,x3) depending on dimension [c/wpe]
        #need norm of magnetic field seen by the macroparticle 'b'

        #get gradient of magnetic field in the cell containing the macroparticle
        #from values on grid
        Bx = self.getB(time, "x")
        By = self.getB(time, "y")
        Bz = self.getB(time, "z")

        if self.ndim==1:
            dx = self.meshSize
            i = self.findCell(pos)   #contains index of cell for each macroparticle

            #len(B[i]) = #parts, can be larger than len(B)
            #corresponds to B in cell with index i for each macroparticle
            kappaX = (Bx[i+1] - Bx[i])/dx*bx
            kappaY = (By[i+1] - By[i])/dx*bx
            kappaZ = (Bz[i+1] - Bz[i])/dx*bx

        elif self.ndim==2:
            dx, dy = self.meshSize
            i, j = self.findCell(pos)

            kappaX = ((Bx[i+1,j  ] - Bx[i,j])/dx*bx +
                      (Bx[i  ,j+1] - Bx[i,j])/dy*by)

            kappaY = ((By[i+1,j  ] - By[i,j])/dx*bx +
                      (By[i  ,j+1] - By[i,j])/dy*by)

            kappaZ = ((Bz[i+1,j  ] - Bz[i,j])/dx*bx +
                      (Bz[i  ,j+1] - Bz[i,j])/dy*by)

        elif self.ndim==3:
            dx, dy, dz = self.meshSize
            i, j, k = self.findCell(pos)

            kappaX = ((Bx[i+1,j  ,k  ] - Bx[i,j,k])/dx*bx +
                      (Bx[i  ,j+1,k  ] - Bx[i,j,k])/dy*by +
                      (Bx[i  ,j  ,k+1] - Bx[i,j,k])/dz*bz)

            kappaY = ((By[i+1,j  ,k  ] - By[i,j,k])/dx*bx +
                      (By[i  ,j+1,k  ] - By[i,j,k])/dy*by +
                      (By[i  ,j  ,k+1] - By[i,j,k])/dz*bz)

            kappaZ = ((Bz[i+1,j  ,k  ] - Bz[i,j,k])/dx*bx +
                      (Bz[i  ,j+1,k  ] - Bz[i,j,k])/dy*by +
                      (Bz[i  ,j  ,k+1] - Bz[i,j,k])/dz*bz)

        norm2 = bx**2+by**2+bz**2

        return kappaX/norm2, kappaY/norm2, kappaZ/norm2


    #--------------------------------------------------------------
    def createTagsFile(self, species, outPath, step=None):

        time = self.getTimeAxis(species,raw=True)
        tag = self.getRaw(time, species, "tag")   #[time,part]

        start=True
        for i in range(len(tag)):

            if start:
                stackedTags = tag[i]
                start=False
            else:
                cond = np.isin(tag[i],stackedTags)   #false if tag is NOT already set
                keep = ~np.logical_and(cond[:,0],cond[:,1])
                #add tag if both node and particle number are not already in
                stackedTags = np.vstack((stackedTags, tag[i][keep]))

        #sort the tags
        stackedTags=sorted(stackedTags, key=operator.itemgetter(0, 1))[::step]
        # stackedTags = stackedTags[::step]
        print("Species",species+":",len(stackedTags),"tags")

        with open(outPath,'w') as f:

            # First line of file should contain the total number of tags followed by a comma
            f.write(str(len(stackedTags))+',\n')
            # The rest of the file is just the node id and particle id for each tag, each followed by a comma
            for node_id,particle_id in stackedTags:
                f.write(str(node_id)+', '+str(particle_id)+',\n')

        return


    #--------------------------------------------------------------
    def writeHDF5(self, data, name, time_series=True):

        dataPath = self.path+"/PP/"+name

        #make sure folder exists
        if not os.path.exists(dataPath): os.makedirs(dataPath)

        #multiple files
        if time_series:
            for i in range(len(data)):

                filePath = dataPath+"/"+name+"-"+str(i).zfill(6)+".h5"

                with h5py.File(filePath, 'w') as hf:
                    hf.create_dataset(name, data=data[i])

        #single file
        else:
            filePath = dataPath+"/"+name+".h5"

            with h5py.File(filePath, 'w') as hf:
                hf.create_dataset(name, data=data)

        return


    #--------------------------------------------------------------
    def getTrackData(self, species, key, sl=slice(None)):

        dataPath = self.path+"/MS/TRACKS/"+species
        filePath = dataPath+"/"+key+".h5"

        #check whether track data was already ordered
        if not os.path.exists(filePath):

            if not os.path.exists(dataPath): os.makedirs(dataPath)

            #read unordered data, order it and dump to disk for next time
            #first dimension is macroparticles, second is time
            data = tr.readUnorderedTrackData(self.path, species, key)

            #usually many time steps but few macroparticles
            #dump everything to single file is fine
            with h5py.File(filePath, 'w') as hf:
                hf.create_dataset(key, data=data)

        else:
            #load ordered data, much faster
            with h5py.File(filePath,"r") as f:
                data = f[key][sl]

        return data



