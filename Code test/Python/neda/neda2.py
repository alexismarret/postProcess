#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:50:09 2022

@author: alexis
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

params={'axes.titlesize' : 12, 'axes.labelsize' : 12, 'lines.linewidth' : 1.8,
        'lines.markersize' : 2, 'xtick.labelsize' : 12, 'ytick.labelsize' : 12,
        'font.size': 10,'legend.fontsize': 12, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True}
plt.rcParams.update(params)

data = np.loadtxt("NNT_activity.txt",unpack=True)
x  = data[0]
y1 = data[1]
y2 = data[2]
y3 = data[3]
y4 = data[4]

R=len(x)
st=5.     #standard deviation for gaussian filter
#======================================
#gaussian filter

win=signal.gaussian(R,st)   #window
smoothed_y1 = signal.convolve(y1, win, mode='same') / sum(win)
smoothed_y2 = signal.convolve(y2, win, mode='same') / sum(win)
smoothed_y3 = signal.convolve(y3, win, mode='same') / sum(win)
smoothed_y4 = signal.convolve(y4, win, mode='same') / sum(win)


#======================================
plt.close("all")
fig, ((sub1,sub2),(sub3,sub4)) = plt.subplots(2,2,figsize=(4.1,2),dpi=300,sharex=True)
plt.gcf().subplots_adjust(bottom=0.15)

sub1.plot(x,y1/y1[0],color="b")
sub1.plot(x,smoothed_y1/y1[0],color="r")

sub2.plot(x,y2/y2[0],color="b")
sub2.plot(x,smoothed_y2/y2[0],color="r")

sub3.plot(x,y3/y3[0],color="b")
sub3.plot(x,smoothed_y3/y3[0],color="r")

sub4.plot(x,y4/y4[0],color="b")
sub4.plot(x,smoothed_y4/y4[0],color="r")

sub1.plot(x,0.005/60.*x+1.,color="gray",linestyle="--")
sub2.plot(x,0.005/60.*x+1.,color="gray",linestyle="--")
sub3.plot(x,0.005/60.*x+1.,color="gray",linestyle="--")
sub4.plot(x,0.005/60.*x+1.,color="gray",linestyle="--")

mi=0.85
ma=1.15
sub1.set_ylim(mi,ma)
sub2.set_ylim(mi,ma)
sub3.set_ylim(mi,ma)
sub4.set_ylim(mi,ma)

sub1.set_title("series 1")
sub2.set_title("series 2")
sub3.set_title("series 3")
sub4.set_title("series 4")
# sub1.legend(loc='upper right')

sub3.set_xlabel("Time (s)")
sub4.set_xlabel("Time (s)")