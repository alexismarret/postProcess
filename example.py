#!/usr/bin/env python3


import osiris     #import osiris class

#------------------------------------
run  ="test_run"   #name of the directory containing the run

o = osiris.Osiris(run)    #call the osiris class

#------------------------------------
time = o.getTimeAxis()     #get the times (default is of EM fields diagnostic)
x = o.getAxis("x")         #get the x axis grid coordinates

Uix = o.getUfluid(time, "ion", "x")     #get the momentum of species "ion" in direction "x"

