#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:33:13 2022

@author: alexis
"""

import osiris
import os
import numpy as np

#----------------------------------------------
run = "CS3DtagsRetrieve"
o = osiris.Osiris(run)

species=["eL","iL"]
step = 1000

#----------------------------------------------
for i in range(len(species)):

    outPath = o.path+"/"+run+"_"+species[i]+".tags"

    o.createTagsFile(species[i],outPath,step=step)
