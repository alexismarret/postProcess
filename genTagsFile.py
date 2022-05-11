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
run = "CS2Drmhr"
o = osiris.Osiris(run)

species=["eL","eR","iL","iR"]
step = 20

#----------------------------------------------
outPath = o.path+"/tags"
if not os.path.exists(outPath):
    os.makedirs(outPath)
else:
    for file in os.listdir(outPath): os.remove(outPath+"/"+file)

#----------------------------------------------
for i in range(len(species)):
        o.createTagsFile(species[i],outPath+"/"+species[i]+".tags",step=step)
