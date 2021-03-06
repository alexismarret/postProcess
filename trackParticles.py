#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:11:22 2022

@author: alexis
"""

import h5py
import numba
import numpy as np


#--------------------------------------------------------------
def readUnorderedTrackData(path, species, qty):
    # Read chunked hdf5 track data and order it correctly.
    # This works because of the 'itermap' member. Every row of itermap contains information about the next
    # several rows of the 'data' attribute. Each row of itermap is (itrack, chunklen, isimulation) where
    # itrack is the particle track id for the next chunklen rows of the 'data' member, and begins at sim step
    # isimulation. The algorithm works by iterating through itermap and pasting each chunklen rows of 'data' into
    # the output slot for that particle. It builds up the output for each particle until returning at the end.
    #
    # If particles are gained/lost during the simulation, their tracks will have unequal lengths and so the
    # remaining space in the output is padded with zeros.
    #
    # Input: Path of simulation data, species name
    # Output: a dict containing the track data. Keys are ['t', 'q', 'ene', 'x1', 'x2', 'x3', 'p1', 'p2', 'p3']
    #    If you enabled ifdmp_tracks_efl or ifdmp_tracks_bfl, it includes ['E1', 'E2', 'E3', 'B1', 'B2', 'B3']
    #    Each element of the dict is a 2D array with dimensions (particle_index, timestep)

    with h5py.File(path+'/MS/TRACKS/'+species+'-tracks.h5', 'r') as tracks:

        labels = [s.decode() for s in tracks.attrs['QUANTS']]  # Get quant labels as regular (non-binary) strings
        ntracks = tracks.attrs['NTRACKS'][0]  #number of individual macroparticles tracked
        itermap = tracks['itermap'][:] # Get itermap as a numeric array

        data = tracks['data'][:,labels.index(qty)-1] # Get the data as a numeric array

    data = reorderTableNumba(data, itermap, ntracks)

    return data


#--------------------------------------------------------------
@numba.njit() # This is a faster version of reorderTable using Numba. You can comment this line out if you don't want to use Numba.
def reorderTableNumba(data, itermap, ntracks):

    ntracks_times_nsteps = data.shape[0]
    nsteps = ntracks_times_nsteps//ntracks
    ntracks = np.uint64(ntracks) # The hdf5 file gives a 32-bit int but Numba wants 64-bit

    output = np.zeros((ntracks,nsteps), dtype=data.dtype) # Allocate assuming equal track lengths
    ioutput_all = np.zeros(ntracks, dtype=np.int64) # Keep a cursor for writing each output column

    idata = 0 # Which row of the data we are processing

    for i_itermap in range(itermap.shape[0]): # itrack is i(t0) for this chunk and particle
        itrack, chunklen, isimulation = itermap[i_itermap,:] # Next row of the itermap
        ioutput = ioutput_all[itrack-1] # Output data cursor location for this quant and particle

        if ioutput+chunklen > output.shape[1]: # If output needs to be bigger, make it bigger
            nt, ns = output.shape
            newOutput = np.zeros((nt,int(np.round(ns*1.5))),dtype=output.dtype) # Make a bigger output array
            newOutput[:,:ns] = output # Copy data over into bigger array
            output = newOutput # Switch over to the new output array

        output[itrack-1,ioutput:ioutput+chunklen] = data[idata:idata+chunklen] # Paste the data

        idata += chunklen # Set cursor past the pasted data in the 'data' attribute
        ioutput_all[itrack-1] += chunklen # Set cursor to the next open row in the output

    return output



