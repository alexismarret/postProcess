#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 19:22:09 2022

@author: alexis
"""


#----------------------------------------------
run  ="counterStreamFast"
o = osiris.Osiris(run,spNorm="iL")

sx = slice(None,None,1)
st = slice(None,None,10)
x    = o.getAxis("x")[sx]
y    = o.getAxis("y")[sx]
time = o.getTimeAxis("iL")[st]


UeL = np.stack((o.getUfluid(time, "iL","x"),
                o.getUfluid(time, "iL","y"),
                o.getUfluid(time, "iL","z")),axis=-1)

UeL = np.stack((o.getUfluid(time, "iL","x"),
                o.getUfluid(time, "iL","y"),
                o.getUfluid(time, "iL","z")),axis=-1)

