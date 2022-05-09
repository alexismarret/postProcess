#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:25:20 2022

@author: alexis
"""


import plots3d as p3D
import numpy as np
import osiris
import matplotlib.pyplot as plt
plt.close("all")
#----------------------------------------------
run  ="CS3D"
o = osiris.Osiris(run,spNorm="iL")
muFac = 1/np.sqrt(o.rqm[o.sIndex("iL")])

st=slice(0,None,1)
# st=slice(0,2,1)
time = o.getTimeAxis()[st]

#----------------------------------------------
o.setup_dir(o.path+"/plots/3Dj1")
image=True
pixels = (256,256)

xlim=np.array([0,10])/muFac     #c/wpe
ylim=np.array([0,10])/muFac
zlim=np.array([0,10])/muFac

#----------------------------------------------
#CS3D
# rcamera=np.array([-600,-380,430])   #c/wpe
rcamera=np.array([-590,-390,420])*muFac  #c/wpi
fov = (.45,.45)
thcamera=np.array([35,20])

#test3DdumpRawTrack
"""
rcamera=np.array([-20,-13,11])
fov = (.45,.45)
thcamera=np.array([35,+20])
"""

#----------------------------------------------
for i in range(0,len(time)):
    fig = p3D.plotPathTrace([o.path+'/MS/FLD/j1'],['j1'],i,
                  rcamera=rcamera,
                  thcamera=thcamera,
                  pixels=pixels,
                  image=image,
                  vlabel=r"$J_x\ [en_ec]$",
                  # xlim=xlim,ylim=ylim,zlim=zlim,
                  dr=0.5,
                  figsize=(5,4),dpi=200,opacity=0.3,    #opacity : physical length x opacity x value of pixel =1
                  vlim=(-0.1,0.1),cmap='bwr',fov=fov,
                  cbarloc='center right',
                  tickparams=([[10,5],[10,5],[10,5]]),
                  scale_axes=([muFac,muFac,muFac]))

    plt.savefig(o.path+"/plots/3Dj1"+"/plot-{i}-time-{t}.png".format(i=i,t=time[i]),dpi="figure")
    plt.close()




#----------------------------------------------
"""
pixels = (np.array([300,300])).astype(int)
frame = 2
slices = np.array([0,1000])
rcamera=np.array([-156,400,400])
fov = (.7,.7)
thcamera=np.array([-45,45])


frame = 2
x0 = 0
f3D.plotPathTrace([froot+'/MS/FLD/j1'],['j1'],frame,
                    rcamera=rcamera,thcamera=thcamera,pixels=pixels,
                    dr=0.5,
                    figsize=(4,4),dpi=300,opacity=0.1,
                    # xslices=slices,
                    # combineFunc=lambda v:-v[0],
                    vlim=(-0.1,0.1),cmap='jet',fov=fov,0
                    show=True,axesback=True,axesfront=True,image=True,
                    # xlim=[0,50],ylim=[0,50],zlim=[0,50],
                    vlabel='$n_b~(n_0)$',
                    # offsets=[x0,0,0],scales=[1,1,1],
                    cbarloc='center right')
"""
#opacity : physical length x opacity x value of pixel =1
