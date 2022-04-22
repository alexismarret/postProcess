#!/usr/bin/env python3


import osiris     #import osiris class
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------
run  ="counterStreamFast"   #name of the directory containing the run

o = osiris.Osiris(run)    #call the osiris class

#------------------------------------
time = o.getTimeAxis()     #get the times (default is of EM fields diagnostic)
x = o.getAxis("x")         #get the x axis grid coordinates
y = o.getAxis("y")
Uix = o.getUfluid(time, "iL", "x")     #get the momentum of species "ion" in direction "x"



fig, (sub1) = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(time,np.mean(Uix,axis=(1,2)))

