#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 16:46:57 2022

@author: alexis
"""
import numpy as np
from scipy.optimize import leastsq

#==================================================================
#fit exponential : Y = amp*exp(X*index)

def fitExponential (X, Y, pinit=[0,0]) :


    fitfunc=lambda p,x : p[0]*np.exp(x*p[1])
    errfunc=lambda p,x,y : y-fitfunc(p,x)

    out,cov,infodict,mesg,ier = leastsq(errfunc,pinit,args=(X,Y),full_output=True)

    amp=out[0]
    index=out[1]

    # ss_err=(infodict['fvec']**2).sum()
    # ss_tot=((Y-Y.mean())**2).sum()
    # rsquared = ss_err/ss_tot


    return amp, index
