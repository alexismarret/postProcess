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
    def __init__(self, run, spNorm="", nbrCores=1):

        self.path = os.environ.get("OSIRIS_RUN_DIR") + "/" + run
        self.nbrCores = nbrCores
        self.spNorm = spNorm

        self.parseInput()

        return

    #--------------------------------------------------------------
    def parseInput(self):

        #open input file
        with open(glob.glob(self.path+"/*.in")[0]) as f:

            Ns=0
            s=0
            cat=""

            for l in f.readlines():

                #remove bracket, spaces, line breaks
                l = l.replace("{","").replace(" ","").replace("\n","")

                #skip comment lines
                if l[0]=="!": continue

                #find needed categories
                if (l=="diag_emf") or (l=="diag_species"): cat = l

                #filter for numerical inputs
                elif "=" in l:
                    #remove d0 notation, quote and last comma
                    l  = l.replace("d0","").replace('"','')[:-1]
                    value = l[l.index("=")+1:]

                    #---------------------------
                    #grid parameters
                    if "nx_p" in l:
                        R = np.int_(value.split(","))
                        self.Nx = R[0]
                        if len(R)==1:
                            self.grid = (self.Nx,)
                        elif len(R)==2:
                            self.Ny   = R[1]
                            self.grid = (self.Nx,self.Ny)
                        else:
                            self.Ny   = R[1]
                            self.Nz   = R[2]
                            self.grid = (self.Nx,self.Ny,self.Nz)

                    elif "xmin" in l:
                        R = np.float_(value.split(","))
                        self.xmin = R[0]
                        if len(R)==2:
                            self.ymin = R[1]
                        elif len(R)==3:
                            self.ymin = R[1]
                            self.zmin = R[2]

                    elif "xmax" in l:
                        R = np.float_(value.split(","))
                        self.xmax = R[0]
                        if len(R)==2:
                            self.ymax = R[1]
                        elif len(R)==3:
                            self.ymax = R[1]
                            self.zmax = R[2]

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
                        self.ndump_fac_rawP = np.zeros(Ns)
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

                    elif (cat=="diag_species") and ("ndump_fac_raw=" in l):
                        self.ndump_fac_rawP[s] = int(value)

        return


    #--------------------------------------------------------------
    def getRatioQM(self, species):

        index=np.nonzero(np.in1d(self.species_name,species))[0]

        rqm = self.rqm[index]

        return rqm


    #--------------------------------------------------------------
    def getAxis(self, direction):

        if   direction == "x":
            delta = (self.xmax - self.xmin) / self.Nx
            axis = np.linspace(self.xmin,self.Nx*delta,self.Nx)

        elif direction == "y":
            delta = (self.ymax - self.ymin) / self.Ny
            axis = np.linspace(self.ymin,self.Ny*delta,self.Ny)

        elif direction == "z":
            delta = (self.zmax - self.zmin) / self.Nz
            axis = np.linspace(self.zmin,self.Nz*delta,self.Nz)

        if self.spNorm!="": axis /= np.sqrt(self.getRatioQM(self.spNorm))


        return axis


    #--------------------------------------------------------------
    def getTimeAxis(self, species="", ene=False, raw=False):

        #species time
        if species!="":
            try: species_index = np.where(self.species_name==species)[0][0]
            except: sys.exit("Unknown species '"+species+"'")

            if ene:
                #retrieve number of dumps from any of the folders in /HIST that is not fld_ene
                file = [x for x in os.listdir(self.path+"/HIST/") if x not in "fld_ene"][0]

                time = np.loadtxt(self.path+"/HIST/"+file,skiprows=2,usecols=1)

            elif raw:
                N = len(os.listdir(self.path+"/MS/RAW/"+species))
                delta = self.dt*self.ndump*self.ndump_fac_rawP[species_index]

                time = np.linspace(self.tmin,(N-1)*delta,N)

            else:
                #retrieve number of dumps from any of the folders in /DENSITY
                N = len(os.listdir(self.path+"/MS/DENSITY/"+species+"/"+
                                   os.listdir(self.path+"/MS/DENSITY/"+species)[0]))
                delta = self.dt*self.ndump*self.ndump_facP[species_index]

                time = np.linspace(self.tmin,(N-1)*delta,N)

        #fields time
        else:
            if ene:
                time = np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=1)

            else:
                #retrieve number of dumps from any of the folders in /FLD
                N = len(os.listdir(self.path+"/MS/FLD/"+
                                   os.listdir(self.path+"/MS/FLD")[0]))
                delta = self.dt*self.ndump*self.ndump_facF

                time = np.linspace(self.tmin,(N-1)*delta,N)

        if self.spNorm!="": time /= np.sqrt(self.getRatioQM(self.spNorm))

        return time.round(7)


    #--------------------------------------------------------------
    def getAxisFourier(self,axis):

        axisF = np.fft.rfftfreq(len(axis),axis[1]-axis[0])*2*np.pi

        return axisF[1], axisF[-1], axisF



    #--------------------------------------------------------------
    def getOnGrid(self, time, dataPath, species=""):

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        #get field or species times
        index=np.nonzero(np.in1d(self.getTimeAxis(species),time))[0]

        #check if requested times exist
        if len(index)!=N: sys.exit("Unknown time for '"+dataPath+"'")

        #parallel reading of data
        it = (dataPath + p for p in np.take(sorted(os.listdir(dataPath)), index))

        G = pf.parallel(pf.readData, it, self.nbrCores)

        return G



    #--------------------------------------------------------------
    def getEnergyIntegr(self, time, qty, species=""):

        #handle list or single time
        try:    N = len(time)
        except: time = [time]; N = 1

        cond=np.in1d(self.getTimeAxis(species,ene=True),time)

        #check if requested times exist
        if len(np.nonzero(cond)[0])!=N: sys.exit("Unknown time for '"+qty+"'")

        #energy per field component
        if qty=="B":
            ene = np.array([np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=2)[cond],
                            np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=3)[cond],
                            np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=4)[cond]],copy=False).T

        elif qty=="E":
            ene = np.array([np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=5)[cond],
                            np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=6)[cond],
                            np.loadtxt(self.path+"/HIST/fld_ene",skiprows=2,usecols=7)[cond]],copy=False).T

        #kinetic energy
        elif qty=="kin":
            sIndex = np.nonzero(np.in1d(self.species_name,species))[0][0] + 1
            #make sure padding is correct
            if sIndex < 10: sIndex = "0" + str(sIndex)
            else          : sIndex = str(sIndex)

            ene = np.loadtxt(self.path+"/HIST/par"+sIndex+"_ene",skiprows=2,usecols=3)[cond]

        elif qty=="T":
            sIndex = np.nonzero(np.in1d(self.species_name,species))[0][0] + 1
            #make sure padding is correct
            if sIndex < 10: sIndex = "0" + str(sIndex)
            else          : sIndex = str(sIndex)

            ene = np.loadtxt(self.path+"/HIST/par"+sIndex+"_temp",skiprows=1,usecols=3)[cond]

        return ene


    #--------------------------------------------------------------
    def getBinit(self):

        initB = np.zeros(self.grid+(3,))

        initB[...,0] = self.ext_b0[0]
        initB[...,1] = self.ext_b0[1]
        initB[...,2] = self.ext_b0[2]

        return initB


    #--------------------------------------------------------------
    def getB(self, time, comp):

        if   comp=="x": key = "b1"
        elif comp=="y": key = "b2"
        elif comp=="z": key = "b3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        B = self.getOnGrid(time,dataPath)

        return B


    #--------------------------------------------------------------
    def getE(self, time, comp):

        if   comp=="x": key = "e1"
        elif comp=="y": key = "e2"
        elif comp=="z": key = "e3"

        dataPath = self.path+"/MS/FLD/"+key+"/"

        E = self.getOnGrid(time,dataPath)

        return E


    #--------------------------------------------------------------
    def getUfluid(self, time, species, comp):

        if   comp=="x": key = "ufl1"
        elif comp=="y": key = "ufl2"
        elif comp=="z": key = "ufl3"

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Ufluid = self.getOnGrid(time,dataPath,species)

        return Ufluid


    #--------------------------------------------------------------
    def getUth(self, time, species, comp):

        if   comp=="x": key = "uth1"
        elif comp=="y": key = "uth2"
        elif comp=="z": key = "uth3"

        dataPath = self.path+"/MS/UDIST/"+species+"/"+key+"/"

        Uth = self.getOnGrid(time,dataPath,species)

        return Uth


    #--------------------------------------------------------------
    def getCharge(self, time, species, cellAv=False):

        """
        Get species charge density C = n*q
        cellAv: moment obtained from macroparticles in the cell, no interpolation
        """

        key = "charge"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Chr = self.getOnGrid(time,dataPath,species)

        return Chr


    #--------------------------------------------------------------
    def getMass(self, time, species, cellAv=False):

        """
        Get species mass density M = n*m
        """

        key = "m"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        M = self.getOnGrid(time,dataPath,species)

        return M


    #--------------------------------------------------------------
    def getCurrent(self, time, species, comp, cellAv=False):

        if   comp=="x": key = "j1"
        elif comp=="y": key = "j2"
        elif comp=="z": key = "j3"

        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Cur = self.getOnGrid(time,dataPath,species)

        return Cur


    #--------------------------------------------------------------
    #get species kinetic energy density
    def getKinEnergy(self, time, species, cellAv=False):

        key = "ene"
        if cellAv: dataPath = self.path+"/MS/CELL_AVG/"+species+"/"+key+"/"
        else:      dataPath = self.path+"/MS/DENSITY/" +species+"/"+key+"/"

        Enrgy = self.getOnGrid(time,dataPath,species)

        return Enrgy


    #--------------------------------------------------------------
    def getPhaseSpace(self, time, species, direction, comp):

        if    direction=="x": l = "x1"
        elif  direction=="y": l = "x2"
        elif  direction=="z": l = "x3"

        if   comp=="x": p = "p1"
        elif comp=="y": p = "p2"
        elif comp=="z": p = "p3"
        elif comp=="g": p = "gamma"

        dataPath = self.path+"/MS/PHA/"+p+l+"/"+species+"/"

        Pha = self.getOnGrid(time,dataPath,species)

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
