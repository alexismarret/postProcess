#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:10:26 2022

@author: alexis
"""

import glob
import os
import sys
import numpy as np
import parallel_functions as pf


class Osiris:

    #--------------------------------------------------------------
    def __init__(self, run, spNorm=None, nbrCores=6):

        self.path = os.environ.get("OSIRIS_RUN_DIR") + "/" + run
        self.allRuns = np.sort(os.listdir(os.environ.get("OSIRIS_RUN_DIR")))
        self.nbrCores = nbrCores
        self.spNorm = spNorm

        self.parseInput()

        self.cellHyperVolume = np.prod((self.gridPosMax-self.gridPosMin)/self.grid)
        self.boxHyperVolume  = np.prod (self.gridPosMax-self.gridPosMin)

        return


    #--------------------------------------------------------------
    def parseInput(self):

        try:
            input_file = glob.glob(self.path+"/*.in")[0]
        except IndexError:
            raise ValueError("Cannot find input file in '"+self.path+"'")

        Ns=0
        s=0
        cat=""

        #open input file
        with open( input_file) as f:
            for l in f.readlines():

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
                    l  = l.replace("d0","").replace('"','')

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

                    elif (cat=="diag_emf") and ("ndump_fac=") in l:
                        self.ndump_facF = int(value)

                    #---------------------------
                    #particles parameters
                    elif ("num_species=" in l) or ("num_cathode=" in l):
                        Ns+=int(value)
                        self.ndump_facP     = np.zeros(Ns)
                        self.ndump_fac_ene  = np.zeros(Ns)
                        self.ndump_fac_raw  = np.zeros(Ns)
                        self.species_name   = np.empty(Ns, dtype='object')
                        self.rqm            = np.zeros(Ns)

                    elif "name=" in l:
                        if self.species_name[s] != None: s+=1
                        self.species_name[s] = value

                    elif "rqm=" in l:
                        if self.rqm[s] != 0: s+=1
                        self.rqm[s] = float(value)

                    elif (cat=="diag_species") and ("ndump_fac=" in l):
                        self.ndump_facP[s] = int(value)

                    elif (cat=="diag_species") and ("ndump_fac_ene=" in l):
                        self.ndump_fac_ene[s] = int(value)

                    elif (cat=="diag_species") and ("ndump_fac_raw=" in l):
                        self.ndump_fac_raw[s] = int(value)

                    elif (cat=="diag_current") and ("ndump_fac=" in l):
                        self.ndump_facC = int(value)

        return


    #--------------------------------------------------------------
    def printAttributes(self):

        print("-------------------------------")
        print("path =", self.path)
        print("allRuns =", self.allRuns)

        print("-------------------------------")
        print("grid =", self.grid)
        print("gridPosMin =", self.gridPosMin)
        print("gridPosMax =", self.gridPosMax)
        print("cellHyperVolume =", self.cellHyperVolume)
        print("boxHyperVolume =", self.boxHyperVolume)

        print("-------------------------------")
        print("dt =", self.dt)
        print("tmin =", self.tmin)
        print("tmax =", self.tmax)
        try: print("ext_b0 =", self.ext_b0)
        except: pass

        print("-------------------------------")
        print("species_name =", self.species_name)
        print("rqm =", self.rqm)
        print("ndump_facP =", self.ndump_facP)
        print("ndump_fac_ene =", self.ndump_fac_ene)
        print("ndump_fac_raw =", self.ndump_fac_raw)

        print("-------------------------------")
        print("ndump =", self.ndump)
        print("ndump_fac_ene_int =", self.ndump_fac_ene_int)
        print("ndump_facF =", self.ndump_facF)
        print("ndump_facC =", self.ndump_facC)

        return


    #--------------------------------------------------------------
    def getRatioQM(self, species):

        index=np.nonzero(np.in1d(self.species_name,species))[0]

        rqm = self.rqm[index][0]

        return rqm


    #--------------------------------------------------------------
    def getAxis(self, direction):

        if   direction == "x": i=0
        elif direction == "y": i=1
        elif direction == "z": i=2

        delta = (self.gridPosMax[i] - self.gridPosMin[i]) / self.grid[i]
        axis = np.linspace(self.gridPosMin[i],(self.grid[i]-1)*delta,self.grid[i])

        if self.spNorm!=None: axis /= np.sqrt(self.getRatioQM(self.spNorm))

        return axis


    #--------------------------------------------------------------
    def getTimeAxis(self, species=None, ene=False, raw=False):

        #species time
        if species!=None:
            try: species_index = np.where(self.species_name==species)[0][0]
            except: raise ValueError("Unknown species '"+species+"'")

            if ene:
                sIndex = np.nonzero(np.in1d(self.species_name,species))[0][0] + 1
                #make sure padding is correct
                if sIndex < 10: sIndex = "0" + str(sIndex)
                else:           sIndex = str(sIndex)
                N = len(np.loadtxt(self.path+"/HIST/par"+sIndex+"_ene",skiprows=2,usecols=0))
                delta = self.dt*self.ndump*self.ndump_fac_ene[species_index]

            elif raw:
                #use glob.glob to ignore hidden files
                N = len(glob.glob(self.path+"/MS/RAW/"+species+"/*"))
                delta = self.dt*self.ndump*self.ndump_fac_raw[species_index]

            else:
                #retrieve number of dumps from any of the folders in /DENSITY
                N = len(glob.glob(self.path+"/MS/UDIST/"+species+"/"+
                                   os.listdir(self.path+"/MS/UDIST/"+species)[0]+"/*"))
                delta = self.dt*self.ndump*self.ndump_facP[species_index]

        #fields time
        else:
            if ene:
                N = len(np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=0))
                delta = self.dt*self.ndump*self.ndump_fac_ene_int

            else:
                #retrieve number of dumps from any of the folders in /FLD
                N = len(glob.glob(self.path+"/MS/FLD/"+
                                   os.listdir(self.path+"/MS/FLD")[0]+"/*"))
                delta = self.dt*self.ndump*self.ndump_facF

        time = np.linspace(self.tmin,(N-1)*delta,N)

        if self.spNorm!=None: time /= np.sqrt(self.getRatioQM(self.spNorm))

        return time.round(7)



    #--------------------------------------------------------------
    def getOnGrid(self, time, dataPath, species, sl, av, parallel):

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        #get field or species times
        index=np.nonzero(np.in1d(self.getTimeAxis(species),time))[0]

        #check if requested times exist
        if len(index)!=N: raise ValueError("Unknown time for '"+dataPath+"'")

        #complete slice if needed
        slices = [slice(None)]*len(self.grid)
        if type(sl) in {slice,int}: sl = (sl,)
        for k,s in enumerate(sl): slices[k]=s

        #invert slices because of needed transposition
        slices = tuple(slices)[::-1]

        #adjust axis average, reversed because of transposition
        #av input can be 1,2,3 corresponding to spatial axis only
        if av!=None:
            if type(av)==int: av = (av,)
            av = self.revertAx((a-1 for a in av))

        #create inputs
        it = ((dataPath + p, slices, av) for p in np.take(sorted(os.listdir(dataPath)), index))

        #parallel reading of data
        if parallel:
            G = pf.parallel(pf.readData, it, self.nbrCores, plot=False)

        #sequential read
        else:
            #calculate size of sliced array, invert again slices and averaged
            #axis to order after transposition
            G = np.zeros((N,)+self.getSlicedSize(slices[::-1],self.revertAx(av)))
            for i in range(N):
                G[i] = pf.readData(next(it)[0], slices, av)

        return G


    #--------------------------------------------------------------
    def revertAx(self,a):

        if a==None:
            return self.grid
        else:
            val = len(self.grid)-1
            a=list(a)
            for i in range(len(a)):
                if   a[i] == 0: a[i] = val
                elif a[i] == val: a[i] = 0

            return tuple(a)


    #--------------------------------------------------------------
    def getSlicedSize(self, sl, av):

        #array containing final indices and step of sliced array
        #None if no slicing in a given direction, and 0 if only one element
        ind = [None]*len(self.grid)

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
            ene = np.asarray([np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=2)[cond],
                              np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=3)[cond],
                              np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=4)[cond]]).T


        elif qty=="E":
            ene = np.asarray([np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=5)[cond],
                              np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=6)[cond],
                              np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=7)[cond]]).T

        #kinetic energy
        elif qty=="kin":
            sIndex = np.nonzero(np.in1d(self.species_name,species))[0][0] + 1
            #make sure padding is correct
            if sIndex < 10: sIndex = "0" + str(sIndex)
            else:           sIndex = str(sIndex)

            ene = np.loadtxt(self.path+"/HIST/par"+sIndex+"_ene",skiprows=2,usecols=3)[cond]

        return ene / self.boxHyperVolume



    #--------------------------------------------------------------
    def getB(self, time, comp, sl=slice(None), av=None, parallel=True):

        if   comp=="x": key = "b1"
        elif comp=="y": key = "b2"
        elif comp=="z": key = "b3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        B = self.getOnGrid(time,dataPath,None,sl,av,parallel)

        return B


    #--------------------------------------------------------------
    def getE(self, time, comp, sl=slice(None), av=None, parallel=True):

        if   comp=="x": key = "e1"
        elif comp=="y": key = "e2"
        elif comp=="z": key = "e3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        E = self.getOnGrid(time,dataPath,None,sl,av,parallel)

        return E


    #--------------------------------------------------------------
    def getUfluid(self, time, species, comp, sl=slice(None), av=None, parallel=True):

        if   comp=="x": key = "ufl1"
        elif comp=="y": key = "ufl2"
        elif comp=="z": key = "ufl3"

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Ufluid = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return Ufluid


    #--------------------------------------------------------------
    def getUth(self, time, species, comp, sl=slice(None), av=None, parallel=True):

        if   comp=="x": key = "uth1"
        elif comp=="y": key = "uth2"
        elif comp=="z": key = "uth3"

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Uth = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return Uth


    #--------------------------------------------------------------
    def getVclassical(self, time, species, comp, sl=slice(None), av=None, parallel=True):

        Ufluid = self.getUfluid(time,species,comp,sl,av,parallel)

        Vclassical = np.sign(Ufluid)*np.sqrt(1./(1./Ufluid**2+1.))

        return Vclassical


    #--------------------------------------------------------------
    def getCharge(self, time, species, cellAv=False, sl=slice(None), av=None, parallel=True):

        """
        Get species charge density C = n*q
        cellAv: moment obtained from macroparticles in the cell, no interpolation
        """

        key = "charge"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Chr = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return Chr


    #--------------------------------------------------------------
    def getMass(self, time, species, cellAv=False, sl=slice(None), av=None, parallel=True):

        """
        Get species mass density M = n*m
        """

        key = "m"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        M = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return M


    #--------------------------------------------------------------
    def getCurrent(self, time, species, comp, cellAv=False, sl=slice(None), av=None, parallel=True):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Cur = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return Cur


    #--------------------------------------------------------------
    def getTotCurrent(self, time, comp, sl=slice(None), av=None, parallel=True):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        totCur = self.getOnGrid(time,dataPath,None,sl,av,parallel)

        return totCur


    #--------------------------------------------------------------
    #get species kinetic energy density
    def getKinEnergy(self, time, species, cellAv=False, sl=slice(None), av=None, parallel=True):

        key = "ene"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Enrgy = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return Enrgy


    #--------------------------------------------------------------
    def getTemp(self, time, species, comp, sl=slice(None), av=None, parallel=True):

        key = "T"+comp

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        T = self.getOnGrid(time,dataPath,species,sl,av,parallel)

        return T


    #--------------------------------------------------------------
    def getPhaseSpace(self, time, species, direction, comp, parallel=True):

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
    def setup_dir(self, path, rm = True):

        if os.path.exists(path) and rm:
            for files in glob.glob(path+"/*.png"):
                os.remove(files)
            for files in glob.glob(path+"/*.eps"):
                os.remove(files)
            for files in glob.glob(path+"/*.pdf"):
                os.remove(files)

        elif not os.path.exists(path):
            os.makedirs(path)

        return


    #--------------------------------------------------------------
    def locFilament(self, time, polarity, fac=2):

        j = (self.getCurrent(time, "eL", "x")+
             self.getCurrent(time, "eR", "x")+
             self.getCurrent(time, "iL", "x")+
             self.getCurrent(time, "iR", "x"))

        #filament defined as j > std(j) initially (except first time step numerical)
        K = polarity * np.std(j[1]) * fac

        #yield true when value is NOT in filament
        if    polarity== 1: mask = np.ma.getmask(np.ma.masked_where(j<K,j,copy=False))
        elif  polarity==-1: mask = np.ma.getmask(np.ma.masked_where(j>K,j,copy=False))

        return mask


    #--------------------------------------------------------------
    def buildUnitVector(self, sh, direction):

        vec = np.zeros(sh)

        if   direction == "x" : vec[...,0] = 1.
        elif direction == "y" : vec[...,1] = 1.
        elif direction == "z" : vec[...,2] = 1.

        return vec


    #--------------------------------------------------------------
    def emfAlignedBasis(self, time, emf) :

        if emf == "B":
            Para = np.stack((self.getB(time, "x"),
                             self.getB(time, "y"),
                             self.getB(time, "z")),axis=-1)

        elif emf == "E":
            Para = np.stack((self.getE(time, "x"),
                             self.getE(time, "y"),
                             self.getE(time, "z")),axis=-1)

        #normal vector
        Normal = np.cross(Para, self.buildUnitVector(Para.shape, direction="y"))
        # Normal = (Fy*ref_vector[...,2] - Fz*ref_vector[...,1] +
        #           Fz*ref_vector[...,0] - Fx*ref_vector[...,2] +
        #           Fx*ref_vector[...,1] - Fy*ref_vector[...,0])

        #perp vector
        Perp = np.cross(Para, Normal)
        # Normal = (Fy*Normal[...,2] - Fz*Normal[...,1] +
        #           Fz*Normal[...,0] - Fx*Normal[...,2] +
        #           Fx*Normal[...,1] - Fy*Normal[...,0])

        #normalization
        np.divide(Para  , np.linalg.norm(Para,   axis=-1, keepdims=True), out = Para)
        np.divide(Normal, np.linalg.norm(Normal, axis=-1, keepdims=True), out = Normal)
        np.divide(Perp,   np.linalg.norm(Perp,   axis=-1, keepdims=True), out = Perp)

        return Para, Normal, Perp


    #--------------------------------------------------------------
    def dot_product(self, A, B):

        return (A[...,0]*B[...,0]+
                A[...,1]*B[...,1]+
                A[...,2]*B[...,2])
