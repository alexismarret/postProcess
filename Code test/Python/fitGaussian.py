#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:20:24 2022

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.close("all")
#----------------------------------------------
n0 = 3.
vDrift = 0.5
T = 1e-1
mp = 100

vth = np.sqrt(T/mp)

X = np.linspace(0.3,0.7,70)

noise = np.random.default_rng().uniform(low=0.7, high=1.3, size=len(X))
Y = n0/(np.sqrt(2*np.pi)*vth) * np.exp(-0.5*((X-vDrift)/vth)**2) * noise

#----------------------------------------------
#gaussian funtion for fit
def maxwellian(X, n, vth, vDrift):

    #vth == sqrt(kBT/m)
    # gauss = n/(np.sqrt(2*np.pi)*vth) * np.exp(-0.5*((X-vDrift)/vth)**2)
    gauss = n * np.exp(-((X-vDrift)/vth)**2)

    return gauss


#----------------------------------------------
ma = max(Y)
p0 = [ma,
      np.std(X),
      X[np.where(Y==ma)[0][0]]]

Fn0, Fvth, FvDrift  = curve_fit(maxwellian, X, Y, p0=p0, maxfev=5000)[0]


#----------------------------------------------
# Maxw      = Fn0  /(np.sqrt(2*np.pi)*Fvth)  * np.exp(-0.5*((X-FvDrift)/Fvth) **2)
guessMaxw = p0[0]/(np.sqrt(2*np.pi)*p0[1]) * np.exp(-0.5*((X-p0[2])  /p0[1])**2)
trueMaxw  = n0   /(np.sqrt(2*np.pi)*vth)   * np.exp(-0.5*((X-vDrift) /vth)  **2)

Maxw      = Fn0   * np.exp(-((X-FvDrift)/Fvth) **2)



#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.axvline(vDrift,color="gray",linestyle="--",linewidth=0.7)
sub1.plot(X,Y,color="k")
sub1.plot(X,trueMaxw,color="g",linestyle="dashdot")

sub1.plot(X,Maxw,color="r",linestyle="dashdot")

sub1.plot(X,guessMaxw,color="b",linestyle="--")

print(p0)
print(n0,vth,vDrift)
print(Fn0, Fvth, FvDrift)
