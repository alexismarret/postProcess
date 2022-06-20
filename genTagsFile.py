#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:33:13 2022

@author: alexis
"""

import osiris
import os
import glob
import numpy as np

#----------------------------------------------
run = "CS2DrmhrTrack"
o = osiris.Osiris(run)

# species=["eL","eR","iL","iR"]
species=["eL"]
step = 20

N_CPU = 64
Tpart = 20
synth = True
#----------------------------------------------
outPath = o.path+"/tags"
if not os.path.exists(outPath):
    os.makedirs(outPath)
else:
    for file in os.listdir(outPath): os.remove(outPath+"/"+file)

#----------------------------------------------
for i in range(len(species)):
        o.createTagsFile(species[i],outPath+"/"+species[i]+".tags",step=step,
                         synth=synth, N_CPU=N_CPU, Tpart=Tpart)
