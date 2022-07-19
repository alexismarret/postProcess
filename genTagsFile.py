#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:33:13 2022

@author: alexis
"""

import osiris
import os

#----------------------------------------------
run = "CS2DrmhrTrack"
o = osiris.Osiris(run)

species=["eL","eR","iL","iR"]
# species=["eL"]
sl = slice(None)   #fraction of tags to track

N_CPU = 1*64
Npart = 1500
synth = True

#----------------------------------------------
outPath = o.path+"/tags"
if not os.path.exists(outPath):
    os.makedirs(outPath)
else:
    for file in os.listdir(outPath): os.remove(outPath+"/"+file)

#----------------------------------------------
for i in range(len(species)):
    o.createTagsFile(species[i],outPath+"/"+species[i]+".tags",sl=sl,
                     synth=synth, N_CPU=N_CPU, Npart=Npart)
