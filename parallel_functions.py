#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:41:52 2022

@author: alexis
"""

import numpy as np
import h5py
from inspect import signature
import matplotlib.pyplot as plt
from multiprocessing import Pool
# import multiprocessing.pool as mpp
from memory_profiler import profile

#--------------------------------------------------------------
# @profile
def parallel(function, it, nbrCores, plot=False):
    #evaluates in parallel all the results, not lazy

    #disable interactive window for plots
    if plot: plt.switch_backend('Agg')

    #generate pool of workers
    with Pool(processes = nbrCores) as pool:

        #single argument for function
        if len(signature(function).parameters)==1:
            results = pool.map(function, it)

        #mutliple arguments for function
        else:
            results = pool.starmap(function, it)

    #kill all processes
    pool.close()
    pool.join()

    #re enable interactive window for plots
    if plot: plt.switch_backend('Qt5Agg')

    #results is in format:
    # [list (times), tuple (nbr of outputs of function), np.array (of various sizes)]
    #for single output:
    # [list (times), np.array]

    if type(results[0])==tuple:
        return (np.asarray(r) for r in zip(*results))
    else:
        return np.asarray(results)


#--------------------------------------------------------------
def readData(dataPath):

    with h5py.File(dataPath,"r") as f:

        #data has inverted space axis, need .T
        return f[list(f.keys())[-1]][()].T #might be wrong in 3D!


#--------------------------------------------------------------
def distrib_task(begin, end, division) :

    #job distribution
    size   = end - begin + 1    #total number of indexes
    segm   = size // division   #number of iteration to perform per division
    remain = size % division    #remaining indexes to calculate after division

    #handles case if less time elements than division
    if segm == 0: division = remain

    #initialization
    lower = begin
    jobs = [[None]*2 for _ in range(division)]

    #loop over divisions
    for i in range(division):

        #distribute indexes into the divisions, accounting for remaining loops
        if remain > 0:
            upper = lower + segm       #upper index, added +1 if additional index to process
            remain -= 1                #keep track of the remaining loops to distribute
        else:
            upper = lower + segm - 1   #upper index if no additional index to process

        jobs[i][0] = lower
        jobs[i][1] = upper + 1

        lower = upper + 1              #next pair lower index

        #next division

    return jobs


