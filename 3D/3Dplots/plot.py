#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:25:20 2022

@author: alexis
"""


import func3Dplots as f3D
import numpy as np
import osiris

#----------------------------------------------
run  ="CS3D"
o = osiris.Osiris(run,spNorm="iL")

froot = o.path

pixels = (np.array([10,10])).astype(int)
frame = 2
slices = np.array([0,1000])
rcamera=np.array([-256,128,128])
fov = (.5,.5)
thcamera=np.array([0,0])


frame = 2
x0 = 0
f3D.plotPathTrace([froot+'/MS/FLD/j1'],['j1'],frame,
                   rcamera=rcamera,thcamera=thcamera,pixels=pixels,
                   dr=0.1,
                   figsize=(4,4),dpi=300,opacity=1,
                   xslices=slices,
                   combineFunc=lambda v:-v[0],
                   vlim=(-0.1,0.1),cmap='bwr',fov=fov,
                   show=True,axesback=True,axesfront=True,image=True,
                   xlim=[0,50],ylim=[0,50],zlim=[0,50],
                   vlabel='$n_b~(n_0)$',
                   offsets=[x0,0,0],scales=[1,1,1],
                   cbarloc='center right',save='')

