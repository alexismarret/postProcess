#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:20:24 2022

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
plt.close("all")

amp = 8000
drift = 0.5
index = 1e3

X = np.linspace(0.3,0.7,1000)

Y = amp*np.exp(-0.5*index*(X-drift)**2)


def gaussian(X, amp, index, drift):


    return amp*np.exp(-0.5*index*(X-drift)**2)

Famp, Findex, Fdrift  = curve_fit(gaussian, X, Y, p0=[1,1,1], maxfev=5000)[0]

maxw = Famp*np.exp(-0.5*Findex*(X-Fdrift)**2)


fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axvline(drift,color="gray",linestyle="--",linewidth=0.7)
sub1.plot(X,Y)

sub1.plot(X,maxw,color="r")
