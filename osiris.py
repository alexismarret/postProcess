#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:10:26 2022

@author: alexis
"""

import glob
import os

import numpy as np

import h5py
import parallelFunctions as pf

from scipy import signal
import pyfftw

# import time as ti
# from memory_profiler import profile



class Osiris:

    #--------------------------------------------------------------
    def __init__(self, run, spNorm=None, globReduced=False, nbrCores=6):

        self.path = os.environ.get("OSIRIS_RUN_DIR") + "/" + run
        self.allRuns = np.sort(os.listdir(os.environ.get("OSIRIS_RUN_DIR")))
        self.nbrCores = nbrCores
        self.globReduced = globReduced

        self.parseInput(run)

        try:    self.normFactor = np.sqrt(np.abs(self.rqm[self.sIndex(spNorm)]))
        except: self.normFactor = 1.

        self.ndim = len(self.grid)
        self.boxSize = self.gridPosMax - self.gridPosMin
        self.meshSize = self.boxSize / self.grid
        self.cellHyperVolume = np.prod(self.meshSize)
        self.boxHyperVolume  = np.prod(self.boxSize)

        self.setup_dir(self.path+"/plots", rm = False)

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
                if   "nx_p" in l: self.grid = np.int_(value.split(","))
                elif "xmin" in l: self.gridPosMin = np.float_(value.split(","))
                elif "xmax" in l: self.gridPosMax = np.float_(value.split(","))

                #---------------------------
                #time parameters
                elif "dt=" in l:    self.dt = float(value)
                elif "ndump=" in l: self.ndump = int(value)
                elif "tmin=" in l:  self.tmin = float(value)
                elif "tmax=" in l:  self.tmax = float(value)

                #---------------------------
                #EM parameters
                elif "init_b0" in l:            self.init_b0 = np.float_(value.split(","))
                elif "ndump_fac_ene_int=" in l: self.ndump_fac_ene_int = int(value)

                #---------------------------
                #particles parameters
                elif ("num_species=" in l) or ("num_cathode=" in l):
                    Ns+=int(value)
                    self.ndump_facP       = np.zeros(Ns)
                    self.ndump_fac_ene    = np.zeros(Ns)
                    self.ndump_fac_raw    = np.zeros(Ns)
                    self.ndump_fac_pha    = np.zeros(Ns)
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

                elif "density=" in l: self.n0[s] = float(value)
                elif "ufl(" in l: self.ufl[s]    = np.float_(value.split(","))
                elif "uth(" in l: self.uth[s]    = np.float_(value.split(","))

                #---------------------------
                #ndump parameters
                elif cat=="diag_species":
                    if   "ndump_fac=" in l:        self.ndump_facP[s] = int(value)
                    elif "ndump_fac_ene=" in l:    self.ndump_fac_ene[s] = int(value)
                    elif "ndump_fac_raw=" in l:    self.ndump_fac_raw[s] = int(value)
                    elif "ndump_fac_pha" in l:     self.ndump_fac_pha[s] = int(value)
                    elif "ndump_fac_tracks=" in l: self.ndump_fac_tracks[s] = int(value)
                    elif "niter_tracks=" in l:     self.niter_tracks[s] = int(value)

                elif (cat=="diag_current") and ("ndump_fac=" in l): self.ndump_facC = int(value)
                elif (cat=="diag_emf")     and ("ndump_fac=" in l): self.ndump_facF = int(value)

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
        try: print("init_b0 =", self.init_b0)
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
        print("ndump_fac_pha =", self.ndump_fac_pha)
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
    def getTimeAxis(self, species=None, ene=False, raw=False, pha=False):

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

            elif pha:
                N = len(glob.glob(self.path+"/MS/PHA/"+
                                   os.listdir(self.path+"/MS/PHA")[0]+"/"+species+"/*"))
                multFactor = self.ndump_fac_pha[species_index]

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

        #adjust axis average, reversed because of transposition
        #av input can be 1,2,3 corresponding to spatial axis only
        if av!=None:
            if type(av)==int: av = (av,)
            av = tuple((a-1 for a in av))
            if transpose: av = self.revertAx(av,slices)

        #invert slices because of needed transposition
        #slices performance can be worse than reading everything
        #does not support list of unordered indices
        if transpose: slices = tuple(slices)[::-1]
        else:         slices = tuple(slices)

        #create inputs
        it = ((dataPath + p, slices, av, transpose) for p in np.take(sorted(os.listdir(dataPath)), index))

        #multiple values read
        if N>1:
            #parallel reading of data
            if parallel:
                G = pf.parallel(pf.readGridData, it, self.nbrCores)
            #sequential reading of data
            else:
                init = True
                for i in range(N):
                    if init:
                        data = pf.readGridData(next(it)[0], slices, av, transpose)
                        G = np.zeros((N,)+data.shape)
                        G[i] = data
                        init = False
                    else:
                        G[i] = pf.readGridData(next(it)[0], slices, av, transpose)

        #single value read
        else:
            G = pf.readGridData(next(it)[0], slices, av, transpose)

        return G


    #--------------------------------------------------------------
    def getRaw(self, time, species, key, parallel=True):

        #key = ['SIMULATION', 'ene', 'p1', 'p2', 'p3', 'q', 'tag', 'x1', 'x2', 'x3']

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

        #multiple values read, potentially very heavy in memory because of possible irregular data shape
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
    def getPhaseSpace(self, time, species, direction, comp, sl=slice(None),
                      parallel=True, transpose=True):

        #[time,position,momentum]

        if    direction=="x": l = "x1"
        elif  direction=="y": l = "x2"
        elif  direction=="z": l = "x3"

        if   comp=="x": p = "p1"
        elif comp=="y": p = "p2"
        elif comp=="z": p = "p3"
        elif comp=="g": p = "gamma"

        key = p+l
        dataPath = self.path+"/MS/PHA/"+key+"/"+species+"/"

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        #get time
        index=np.nonzero(np.in1d(self.getTimeAxis(species,pha=True),time))[0]

        #check if requested times exist
        if len(index)!=N: raise ValueError("Unknown time for '"+dataPath+"'")

        #complete slice if needed
        slices = [slice(None)]*2
        if type(sl) in {slice,int}: sl = (sl,)
        for k,s in enumerate(sl): slices[k]=s

        if transpose: slices = tuple(slices)[::-1]
        else:         slices = tuple(slices)
        av=None

        #create inputs
        it = ((dataPath + p, slices, av, transpose) for p in np.take(sorted(os.listdir(dataPath)), index))

        #multiple values read, very heavy in memory because of irregular data shape
        if N>1:
            #parallel reading of data
            if parallel:
                G = pf.parallel(pf.readGridData, it, self.nbrCores)
            #sequential reading of data
            else:
                init = True
                for i in range(N):
                    if init:
                        data = pf.readGridData(next(it)[0], slices, av, transpose)
                        G = np.zeros((N,)+data.shape)
                        G[i] = data
                        init = False
                    else:
                        G[i] = pf.readGridData(next(it)[0], slices, av, transpose)

        #single value read
        else:
            G = pf.readGridData(next(it)[0], slices, av, transpose)

        return G


    #--------------------------------------------------------------
    def getBoundPhaseSpace(self, species, direction, comp, index=0):

        if    direction=="x": l = "x1"
        elif  direction=="y": l = "x2"
        elif  direction=="z": l = "x3"

        if   comp=="x": p = "p1"
        elif comp=="y": p = "p2"
        elif comp=="z": p = "p3"
        elif comp=="g": p = "gamma"

        key = p+l

        #path to phasespace diag of the species
        dataPath = (self.path+"/MS/PHA/"+key+"/"+species+"/"+
                              np.sort(os.listdir(self.path+"/MS/PHA/"+key+"/"+species))[index])

        with h5py.File(dataPath,"r") as f:
            boundX = f['AXIS']["AXIS1"][()]
            boundY = f["AXIS"]["AXIS2"][()]

        return boundX, boundY


    #--------------------------------------------------------------
    def revertAx(self, av, sl):

        if av==None:
            return (None,)

        else:
            #number of fully sliced direction
            Nav = len([s for s in sl if type(s)==int])

            av = list(av)
            for i in range(len(av)):

                if av[i] == 0:
                    av[i] = self.ndim-1-Nav
                elif (av[i] == 1 and Nav!=0) or av[i] == 2:
                    av[i] = 0

            return tuple(av)


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

        #kinetic energy (thermal and drift)
        elif qty=="kin":
            sIndex = self.sIndex(species) + 1
            #make sure padding is correct
            if sIndex < 10: sIndex = "0" + str(sIndex)
            else:           sIndex = str(sIndex)

            ene = np.loadtxt(self.path+"/HIST/par"+sIndex+"_ene",skiprows=2,usecols=3)[cond] / self.boxHyperVolume

            return ene



    #--------------------------------------------------------------
    def getB(self, time, comp, sl=slice(None), av=None,
             reduced=None, parallel=True, transpose=True):

        if   comp=="x": key = "b1"
        elif comp=="y": key = "b2"
        elif comp=="z": key = "b3"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/FLD/"+key+"/"

        B = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return B


    #--------------------------------------------------------------
    def getE(self, time, comp, sl=slice(None), av=None,
             reduced=None, parallel=True, transpose=True):

        if   comp=="x": key = "e1"
        elif comp=="y": key = "e2"
        elif comp=="z": key = "e3"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/FLD/"+key+"/"

        E = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return E


    #--------------------------------------------------------------
    def getUfluid(self, time, species, comp, sl=slice(None), av=None,
                  reduced=None, parallel=True, transpose=True):

        #get U = p/m = v * lorentz
        if   comp=="x": key = "ufl1"
        elif comp=="y": key = "ufl2"
        elif comp=="z": key = "ufl3"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Ufluid = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Ufluid


    #--------------------------------------------------------------
    def getUth(self, time, species, comp, sl=slice(None), av=None,
               reduced=None, parallel=True, transpose=True):

        #get Uth = sqrt(kB*T/m)
        if   comp=="x": key = "uth1"
        elif comp=="y": key = "uth2"
        elif comp=="z": key = "uth3"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Uth = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Uth


    #--------------------------------------------------------------
    def getCharge(self, time, species, sl=slice(None), av=None,
                  reduced=None, parallel=True, transpose=True):

        """
        Get species charge density C = n*q
        """

        key = "charge"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Chr = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Chr


    #--------------------------------------------------------------
    def getMass(self, time, species, sl=slice(None), av=None,
                      reduced=None, parallel=True, transpose=True):

        """
        Get species mass density M = n*m
        """

        key = "m"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        M = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return M


    #--------------------------------------------------------------
    def getCurrent(self, time, species, comp, sl=slice(None), av=None,
                   reduced=None, parallel=True, transpose=True):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Cur = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Cur


    #--------------------------------------------------------------
    def getTotCurrent(self, time, comp, sl=slice(None), av=None,
                      reduced=None, parallel=True, transpose=True):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/FLD/"+key+"/"

        totCur = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return totCur


    #--------------------------------------------------------------
    #get species kinetic energy density
    def getKinEnergy(self, time, species, sl=slice(None), av=None,
                     reduced=None, parallel=True, transpose=True):

        key = "ene"

        if (reduced==True) or (self.globReduced and reduced!=False): key+="-savg"
        dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Enrgy = self.getOnGrid(time,dataPath,species,sl,av,parallel,transpose)

        return Enrgy


    #--------------------------------------------------------------
    def getNewData(self, time, key, sl=slice(None), av=None,
                   parallel=True, transpose=False):

        dataPath = self.path+"/PP/"+key+"/"

        #data does not need to be transposed back
        D = self.getOnGrid(time,dataPath,None,sl,av,parallel,transpose)

        return D


    #--------------------------------------------------------------
    def setup_dir(self, path, rm = True):

        if os.path.exists(path) and rm:
            for ext in ("png","eps","pdf","mp4"):
                for file in glob.glob(path+"/*."+ext):
                    os.remove(file)

        elif not os.path.exists(path):
            os.makedirs(path)

        return


    #--------------------------------------------------------------
    def crossProduct(self, Ax, Ay, Az, Bx, By, Bz):

        return Ay*Bz-Az*By, Az*Bx-Ax*Bz, Ax*By-Ay*Bx


    #--------------------------------------------------------------
    def projectVec(self, vx, vy, vz, ex, ey, ez, comp):

        #assumes e!= (0,1,0)
        Ix = 0.
        Iy = 1.
        Iz = 0.

        #project vector 'v' in basis 'e' along direction 'comp'
        if comp==0:   #parallel to 'e'
            baseX, baseY, baseZ = ex, ey, ez

        else: #e x unit vector I, perp1 to e
            baseX, baseY, baseZ = self.crossProduct(ex, ey, ez, Ix, Iy, Iz)

            if comp==2: #e x (e x unit vector I), perp2 to e
                baseX[:], baseY[:], baseZ[:] = self.crossProduct(ex, ey, ez, baseX, baseY, baseZ)


        return (vx*baseX + vy*baseY + vz*baseZ) / np.sqrt(baseX**2+baseY**2+baseZ**2)



    #--------------------------------------------------------------
    # @profile
    def helmholtzDecompose(self, time, comp, check=False):

        #https://github.com/shixun22/helmholtz/blob/master/helmholtz.py
        #reuse same data and ftdiv arrays for reduced memory usage
        #computes the compressive (curl E=0, dB/dt=0, E // k) component of the field
        #E_c = IFT[ (FT[E].k) * k / k^2 ]
        #E_r = E - Ec is the rotational component

        #-----------------------------------
        #configure pyfftw and
        flags = ('FFTW_ESTIMATE','FFTW_DESTROY_INPUT')
        nthreads = 4                #4 seems to be the sweet spot
        dtype = 'complex64'         #complex128 is best, but huge memory usage

        timeSeries = (type(time)==np.ndarray)
        if timeSeries: dataShape = np.insert(self.grid,0,len(time))
        else:          dataShape = self.grid

        #prepare aligned unitialized arrays, random values
        data  = pyfftw.empty_aligned(dataShape, dtype, n=nthreads)
        ftdiv = pyfftw.empty_aligned(dataShape, dtype, n=nthreads)

        #plan fft in place along spatial axis
        ax = np.array([-1,-2,-3])[range(self.ndim)]
        fft  = pyfftw.FFTW(data,  data,  axes=ax, direction='FFTW_FORWARD',  flags=flags, threads=nthreads)
        ifft = pyfftw.FFTW(ftdiv, ftdiv, axes=ax, direction='FFTW_BACKWARD', flags=flags, threads=nthreads)

        #-----------------------------------
        dimkx = (self.grid[0],) + (1,)*(self.ndim-1)  #broadcasting from last to first dimension, kx changes along kx axis
        kx = np.fft.fftfreq(self.grid[0],self.meshSize[0]) .reshape(dimkx)  #no need to multiply by 2*np.pi since normalized
        data[:]  = self.getE(time,"x")   #assign field value to real part of data array in place, imaginary part is 0
        ftdiv[:] = fft(data) * kx.astype(dtype)  #fft Ex stored in place in ftdiv array, specify type of kx to avoid copy

        #-----------------------------------
        if self.ndim>1:  #2D
            dimky = (self.grid[1],) + (1,)*(self.ndim-2)
            ky = np.fft.fftfreq(self.grid[1],self.meshSize[1]) .reshape(dimky)
            data[:]  = self.getE(time,"y")
            ftdiv   += fft(data) * ky  #fft Ey, no need to specify type anymore because reason

            #-----------------------------------
            if self.ndim>2:  #3D
                kz = np.fft.fftfreq(self.grid[2],self.meshSize[2])  #no reshape needed, broadcast starts from last dimension
                data[:]  = self.getE(time,"z")
                ftdiv   += fft(data) * kz  #fft Ez

        #handle lower dimensions
        if self.ndim<2: ky = 0.
        if self.ndim<3: kz = 0.

        ftdiv /= (kx**2 + ky**2 + kz**2)

        #null k vector is NaN, set to 0
        #handle multiple times or single one, and dimensionality
        if timeSeries: ftdiv[(slice(None),)+(0,)*self.ndim]=0.
        else:          ftdiv[               (0,)*self.ndim]=0.

        #-----------------------------------
        #check decomposition, memory heavy
        if check:
            Ex = self.getE(time,"x")
            Ey = self.getE(time,"y")
            Ez = self.getE(time,"z")

            Ecx = np.fft.ifftn(ftdiv * kx, axes=ax)
            Ecy = np.fft.ifftn(ftdiv * ky, axes=ax)
            Ecz = np.fft.ifftn(ftdiv * kz, axes=ax)

            Erx = Ex - Ecx
            Ery = Ey - Ecy
            Erz = Ez - Ecz

            i2pi = 1j * 2*np.pi #i in div and rot operator in fourier space and 2pi factor in k
            #-----------------------------------
            divR = np.fft.ifftn((np.fft.fftn(Erx, axes=ax) * kx +
                                 np.fft.fftn(Ery, axes=ax) * ky +
                                 np.fft.fftn(Erz, axes=ax) * kz) * i2pi, axes=ax)

            divC = np.fft.ifftn((np.fft.fftn(Ecx, axes=ax) * kx +
                                 np.fft.fftn(Ecy, axes=ax) * ky +
                                 np.fft.fftn(Ecz, axes=ax) * kz) * i2pi, axes=ax)

            rotR_fx, rotR_fy, rotR_fz = self.crossProduct(kx, ky, kz,
                                                          np.fft.fftn(Erx, axes=ax),
                                                          np.fft.fftn(Ery, axes=ax),
                                                          np.fft.fftn(Erz, axes=ax))
            rotC_fx, rotC_fy, rotC_fz = self.crossProduct(kx, ky, kz,
                                                          np.fft.fftn(Ecx, axes=ax),
                                                          np.fft.fftn(Ecy, axes=ax),
                                                          np.fft.fftn(Ecz, axes=ax))

            rotRx, rotRy, rotRz = (np.fft.ifftn(rotR_fx *i2pi, axes=ax),
                                   np.fft.ifftn(rotR_fy *i2pi, axes=ax),
                                   np.fft.ifftn(rotR_fz *i2pi, axes=ax))
            rotCx, rotCy, rotCz = (np.fft.ifftn(rotC_fx *i2pi, axes=ax),
                                   np.fft.ifftn(rotC_fy *i2pi, axes=ax),
                                   np.fft.ifftn(rotC_fz *i2pi, axes=ax))

            print ('div_rotational max:',  np.max(np.abs(divR)))
            print ('rot_rotational max:',  np.max((np.abs(rotRx),np.abs(rotRy),np.abs(rotRz))))

            print ('div_compressive max:', np.max(np.abs(divC)))
            print ('rot_compressive max:', np.max((np.abs(rotCx),np.abs(rotCy),np.abs(rotCz))))

        #-----------------------------------
        #perform inverse fft, one component per function call to reduce memory usage
        if   comp==0: ifft(ftdiv * kx)
        elif comp==1: ifft(ftdiv * ky)
        elif comp==2: ifft(ftdiv * kz)

        return ftdiv.real


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
    def writeHDF5(self, data, name, timeArray=True, index=0, dtype='float32'):

        dataPath = self.path+"/PP/"+name

        #make sure folder exists
        if not os.path.exists(dataPath): os.makedirs(dataPath)

        #multiple files
        if timeArray:
            for i in range(len(data)):

                filePath = dataPath+"/"+name+"-"+str(i).zfill(6)+".h5"

                with h5py.File(filePath, 'w') as hf:
                    hf.create_dataset(name, data=data[i], dtype=dtype)

        #single file
        else:
            filePath = dataPath+"/"+name+"-"+str(index).zfill(6)+".h5"

            with h5py.File(filePath, 'w') as hf:
                hf.create_dataset(name, data=data, dtype=dtype)

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
            import trackParticles as tr
            data = tr.readUnorderedTrackData(self.path, species, key)

            #usually many time steps but few macroparticles
            #dump everything to single file is fine
            with h5py.File(filePath, 'w') as hf:
                hf.create_dataset(key, data=data)

        #load ordered data, much faster
        with h5py.File(filePath, "r") as f:
            return f[key][sl]


    #--------------------------------------------------------------
    def createTagsFile(self, species, outPath, sl=slice(None),
                       synth=False, N_CPU=None, Npart=None):

        #synth: whether to create the tag file from scratch or from raw data
        #N_CPU: total number of CPU in the simulation, to be used with synth=True
        #Npart: total number of tags wanted, to be used with synth=True

        #tag file from scratch
        if synth:
            import random

            #number of tags per CPU, << Total number
            if Npart <= N_CPU:
                N_CPU = Npart
                N = 1
            else:
                N = Npart // N_CPU

            stackedTags = np.zeros((N*N_CPU,2),dtype=int)
            #maximum tag index should be much smaller than total number
            #of macroparticles per core
            rgeTags = range(1,N*10)  #factor 10 to pick random values different from unity~

            i=0
            for c in range(1,N_CPU+1):
                partTags = random.sample(rgeTags, N)
                for t in partTags:
                    stackedTags[i] = c, t
                    i+=1

        #tag file from raw data
        else:
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
            # import operator
            # stackedTags=sorted(stackedTags, key=operator.itemgetter(0, 1))[sl]
            stackedTags = stackedTags[sl]

        print("Species",species+":",len(stackedTags),"tags saved in",outPath)

        #create tag file
        with open(outPath,'w') as f:

            # First line of file should contain the total number of tags followed by a comma
            f.write(str(len(stackedTags))+',\n')
            # The rest of the file is just the node id and particle id for each tag, each followed by a comma
            for node_id,particle_id in stackedTags:
                f.write(str(node_id)+', '+str(particle_id)+',\n')

        return


    #--------------------------------------------------------------
    def low_pass_filter(self, data, cutoff, axis, res=None, timeSeries=False):

        #low pass filter, modify data in place
        #axis can be 1,2,3 corresponding to spatial axis only
        #0 < cutoff frequency < 1/(2*res)

        if res is None: res = self.meshSize
        for arr in (cutoff,res,axis):
            if type(arr) not in {list,tuple,np.ndarray}: arr = (arr,)

        order = 2

        #filter over axis iteratively
        #indexes could be wrong if array is already sliced and resolution is different along axis
        for i in axis:
            # Get the filter coefficients
            sos = signal.butter(order, Wn = cutoff[i-1],  fs=1./res[i-1],
                                btype='lowpass', output='sos')
            #filter data in place
            if timeSeries: data[:] = signal.sosfiltfilt(sos, data, axis=i)
            else:          data[:] = signal.sosfiltfilt(sos, data, axis=i-1)

        return


    #--------------------------------------------------------------
    def bisection(self, array, value):
        #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        if   (value < array[0]):   return -1
        elif (value > array[n-1]): return n
        jl = 0                       # Initialize lower
        ju = n-1                     # and upper limits.
        while (ju-jl > 1):           # If we are not yet done,
            jm=(ju+jl) >> 1          # compute a midpoint with a bitshift
            if (value >= array[jm]): jl=jm    # and replace either the lower limit
            else: ju=jm   # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
        if   (value == array[0]):   return 0     # edge cases at bottom
        elif (value == array[n-1]): return n-1   # and top
        else:                       return jl


    #--------------------------------------------------------------
    def imExtent(self, x, y):

        #for use with imshow, includes all bins in image
        ext = (x[0], x[-1]+x[1], y[0], y[-1]+y[1])

        return ext



    """
    #--------------------------------------------------------------
    def getSlicedSize(self, sl, av):

        #not needed anymore, just measure array dimension after extracting data

        #array containing final indices and step of sliced array
        #None if no slicing in a given direction, and 0 if only one element
        ind = [None]*len(sl)

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
    """
