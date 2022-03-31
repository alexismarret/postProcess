#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:32:57 2022

@author: alexis
"""

#!/usr/bin/env python3
import numpy as np
import time
from inspect import signature
import matplotlib.pyplot as plt
import tqdm
import numpy as np


"""
import multiprocessing.pool as mpp

#patch starmap into istarmap
#--------------------------------------------------------------
def istarmap(self,func, iterable, chunksize=1):

    # self._check_running()
    # if chunksize < 1:
    #     raise ValueError(
    #         "Chunksize must be 1+, not {0:n}".format(
    #             chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap
from multiprocessing import Pool

#--------------------------------------------------------------
def parallel(function, it, nbrCores, plot=False, *outputs):
    #evaluates in parallel all the results

    #disable interactive window for plots
    if plot: plt.switch_backend('Agg')

    print(function.__name__)

    #generate pool of workers
    with Pool(processes = nbrCores) as pool:

        #single argument for function
        if len(signature(function).parameters)==1:
            results = pool.imap(function, it)

        #mutliple arguments for function
        else:
            results = pool.istarmap(function, it)

        #assign results to outputs, if any
        if outputs==():
            for i, result in enumerate(results):
                pass

        else:
            for i, result in enumerate(results):
                #single variable result from function
                if type(result)!=tuple:
                    outputs[0][i] = result

                #multiple variables result from function
                else:
                    for j, output in enumerate(outputs):
                        output[i] = result[j]


    #re enable interactive window for plots
    if plot: plt.switch_backend('Qt5Agg')

    return

"""

from multiprocessing import Pool
# import multiprocessing.pool as mpp
from memory_profiler import profile

#--------------------------------------------------------------
@profile
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


    print("")
    def dim(a):
        print(type(a))
        if type(a) in {tuple,np.ndarray,list}:
            return [len(a)]+dim(a[0])
        else:
            return []

    print(dim(results))

    #results is in format:
    # [list (times), tuple (nbr of outputs of function), np.array (of various sizes)]
    #for single output:
    # [list (times), np.array]

    if type(results[0])==tuple:
        return (np.asarray(r) for r in zip(*results))
    else:
        return np.asarray(results)

        return results


def func(c,d):

    cc= c+1
    dd= d+1

    return cc

L = 100
l = 200

#inputs
c = np.ones((L,l,l))
d = np.ones((L,l,l*5))

#outputs
rc=np.zeros((L,l,l))
rd=np.zeros((L,l,l*5))

it = ((c[i],d[i]) for i in range(L))

nbrCores = 6

parallel(func, it, nbrCores, True)




"""
def function(x,y,z):

    time.sleep(0.1)

    return 3*x, y**2, z+1

import multiprocessing.pool as mpp

#patch starmap into istarmap
#--------------------------------------------------------------
def istarmap(self,func, iterable, chunksize=1):

    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap
from multiprocessing import Pool


#--------------------------------------------------------------
def parallel(function, it, N, nbrCores, plot=False):
    #evaluates in parallel all the results, not lazy

    #disable interactive window for plots
    if plot: plt.switch_backend('Agg')

    #generate pool of workers
    with Pool(processes = nbrCores) as pool:

        #single argument for function
        if len(signature(function).parameters)==1:
            print("imap")
            results = tuple(tqdm.tqdm(pool.imap(function, it),total=N))

        #mutliple arguments for function
        else:
            print("istarmap")
            results = tuple(tqdm.tqdm(pool.istarmap(function, it),total=N))

    # print(r)
    #kill all processes
    pool.close()
    pool.join()

    print("")
    def dim(a):
        print(type(a))
        if type(a) in {tuple,np.ndarray,list}:
            return [len(a)]+dim(a[0])
        else:
            return []

    dim(results)

    #re enable interactive window for plots
    if plot: plt.switch_backend('Qt5Agg')

    #results is in format:
    # [tuple (times), tuple (nbr of outputs of function), np.array (of various sizes)]
    #for single output:
    # [tuple (times), np.array]

    if type(results[0])==tuple:
        print("multiple output")
        return (np.asarray(r) for r in zip(*results))
    else:
        print("single output")
        return np.asarray(results)

#----------------------------------------------
"""

"""
    #assign results to outputs, if any
    if outputs!=():
        #----------------------------------------------
        #recursive function to get dimensions of array without numpy
        def dim(a):
            if type(a) in {tuple,np.ndarray,list}:
                return [len(a)]+dim(a[0])
            else:
                return []
        #----------------------------------------------
        cond = (dim(outputs)==dim(results))

        for j, output in enumerate(outputs):
            for i, result in enumerate(results):

                #multiple variables result from function
                if cond: output[i] = result[j]
                #single variable result from function
                else:    output[i] = result
"""

"""
    L = len(outputs)
    pool = multiprocessing.Pool(processes = nbrCores)

    try:    results = pool.starmap(function, it)
    except: results = pool.imap   (function, it)

    for j, output in enumerate(outputs):
        for i, result in enumerate(results):

            if len(result) == L: output[i] = result[j]
            else:                output[i] = result

    pool.terminate()
    pool.join()
"""
"""
#inputs
x = np.ones(100)
y = np.ones(100)
z = np.ones(100)

multiplied_x = np.zeros(100)
squared_y = np.zeros(100)
added_z = np.zeros(100)
nbrCores = 5



start = time.time()
parallel(function,zip(x,y,z), len(x), nbrCores, multiplied_x, squared_y,added_z)
print(time.time()-start)
"""


"""
#--------------------------------------------------------------
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
"""




"""
#patch starmap into istarmap
#--------------------------------------------------------------
def istarmap(self,func, iterable, chunksize=1):

    # self._check_running()
    # if chunksize < 1:
    #     raise ValueError(
    #         "Chunksize must be 1+, not {0:n}".format(
    #             chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap
from multiprocessing import Pool



#--------------------------------------------------------------
def parallel(function, it, N, nbrCores, plot=False, *outputs):
    #evaluates in parallel all the results

    #disable interactive window for plots
    if plot: plt.switch_backend('Agg')

    print(function.__name__)

    #generate pool of workers
    with Pool(processes = nbrCores) as pool:

        #single argument for function
        if len(signature(function).parameters)==1:
            results = tqdm.tqdm(pool.imap(function, it),total=N)

        #mutliple arguments for function
        else:
            results = tqdm.tqdm(pool.istarmap(function, it),total=N)

    #re enable interactive window for plots
    if plot: plt.switch_backend('Qt5Agg')

    #results is in format:
    # [tuple (times), tuple (nbr of outputs of function), np.array (of various sizes)]
    #for single output:
    # [tuple (times), np.array]

    if type(results[0])==tuple:
        return (np.asarray(r) for r in zip(*results))
    else:
        return np.asarray(results)

"""
