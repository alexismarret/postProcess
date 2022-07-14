#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:41:20 2022

@author: alexis
"""

#----------------------------------------------
import osiris
import numpy as np

#----------------------------------------------
# run  ="CS3D_noKink"
# run = "CS3Dtrack"
run = 'CS2DrmhrTrack'
# run = "CS3Drmhr"
# run = "Test3D/test3DdumpRaw"

o = osiris.Osiris(run)

st = slice(None,None,1)
time = o.getTimeAxis()[st]

start = 3

timeArray = True
check = False
dtype = "float32"

#----------------------------------------------
#time series
if timeArray:
    compr = o.helmholtzDecompose(comp=0, time=time, check=check)
    o.writeHDF5(compr, "Ecx", timeArray=timeArray, dtype=dtype)

    compr = o.helmholtzDecompose(comp=1, time=time)
    o.writeHDF5(compr, "Ecy", timeArray=timeArray, dtype=dtype)

    compr = o.helmholtzDecompose(comp=2, time=time)
    o.writeHDF5(compr, "Ecz", timeArray=timeArray, dtype=dtype)

#----------------------------------------------
#loop over time
else:
    for i in range(start,len(time)):

        #empty memory before next loop to avoid 2x same array size in temp memory
        print(i)

        compr = o.helmholtzDecompose(comp=0, time=time[i], check=check)
        o.writeHDF5(compr, "Ecx", timeArray=timeArray, index=i)
        del compr

        compr = o.helmholtzDecompose(comp=1, time=time[i])
        o.writeHDF5(compr, "Ecy", timeArray=timeArray, index=i)
        del compr

        compr = o.helmholtzDecompose(comp=2, time=time[i])
        o.writeHDF5(compr, "Ecz", timeArray=timeArray, index=i)
        del compr

