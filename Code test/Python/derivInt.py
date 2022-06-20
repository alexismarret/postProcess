#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:15:03 2022

@author: alexis
"""
#----------------------------------------------
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

#----------------------------------------------
params={'axes.titlesize' : 9, 'axes.labelsize' : 9, 'lines.linewidth' : 1,
        'lines.markersize' : 3, 'xtick.labelsize' : 9, 'ytick.labelsize' : 9,
        'font.size': 9,'legend.fontsize': 9, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
         'figure.autolayout': True,'text.usetex': True}
plt.rcParams.update(params)
plt.close("all")

#----------------------------------------------
x = np.linspace(0,100,101)

dx = x[1]-x[0]

y = np.sin(x)

deriv_y = np.gradient(y,dx,edge_order=2)

I_deriv_x = cumulative_trapezoid(deriv_y,x,initial=0)


#----------------------------------------------
fig, sub1 = plt.subplots(1,figsize=(4.1,2.8),dpi=300)

sub1.plot(x,I_deriv_x,color="k",marker="o")
sub1.plot(x,y,linestyle="--",color="orange",marker="x")
