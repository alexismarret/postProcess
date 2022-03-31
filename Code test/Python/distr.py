#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:28:59 2021

@author: alexis
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")


v = np.linspace(-100, 100,1000)

vtm = 10.

vs = 30.
vts = 50.

fm = np.exp(-v**2/vtm**2)
fs = 0.1*np.exp(-(v-vs)**2/vts**2)


#=================================
#create figure
fig, (sub1) = plt.subplots(1,figsize=(4.1,2),dpi=300,sharex=True)

# sub1.set_ylim(0,1.1)
#plot data
sub1.plot(v,fm)
sub1.plot(v,fs)
sub1.axhline(0,color="k",linewidth=1.5)
# sub1.axvline(0,color="k",linewidth=1.5)

sub1.axis('off')

#arrow on figure
#abscissa
sub1.annotate('', xy=(110 ,0), xytext=(100,0),
              arrowprops=dict(facecolor='black', shrink=0.,
                              width=0.5,headwidth=6,headlength=5))

#ordinate
# sub1.annotate('', xy=(0 ,1.1), xytext=(0,0),
#               arrowprops=dict(facecolor='black', shrink=0.,
#                               width=0.5,headwidth=6,headlength=5))