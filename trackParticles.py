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
def getTrackData(fname):
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
    # Input: Filename of the track data
    # Output: a dict containing the track data. Keys are ['t', 'q', 'ene', 'x1', 'x2', 'p1', 'p2', 'p3']
    #    If you enabled ifdmp_tracks_efl or ifdmp_tracks_bfl, it includes ['E1', 'E2', 'E3', 'B1', 'B2', 'B3']
    #    Each element of the dict is a 2D array with dimensions (particle_index, timestep)

    with h5py.File(fname, 'r') as tracks:
        #print([e for e in tracks.attrs])
        labels = [s.decode() for s in tracks.attrs['QUANTS']] # Get quant labels as regular (non-binary) strings
        data = tracks['data'][:] # Get the data as a numeric array
        itermap = tracks['itermap'][:] # Get itermap as a numeric array
        ntracks = tracks.attrs['NTRACKS'][0]

    table_data = reorderTableNumba(data, itermap, ntracks)
    output_dict = {}
    for i,label in enumerate(labels[1:]):
        output_dict[label] = table_data[i,:,:]

    return output_dict


#--------------------------------------------------------------
@numba.njit() # This is a faster version of reorderTable using Numba. You can comment this line out if you don't want to use Numba.
def reorderTableNumba(data, itermap, ntracks):

    ntracks_times_nsteps, nquants = data.shape
    nsteps = ntracks_times_nsteps//ntracks
    ntracks = np.uint64(ntracks) # The hdf5 file gives a 32-bit int but Numba wants 64-bit

    output = np.zeros((nquants,ntracks,nsteps), dtype=data.dtype) # Allocate assuming equal track lengths
    ioutput_all = np.zeros((nquants,ntracks),dtype=np.int64) # Keep a cursor for writing each output column

    for iquant in range(nquants): # For each reported variable
        idata = 0 # Which row of the data we are processing

        for i_itermap in range(itermap.shape[0]): # itrack is i(t0) for this chunk and particle
            itrack, chunklen, isimulation = itermap[i_itermap,:] # Next row of the itermap
            ioutput = ioutput_all[iquant, itrack-1] # Output data cursor location for this quant and particle

            if ioutput+chunklen > output.shape[2]: # If output needs to be bigger, make it bigger
                nq, nt, ns = output.shape
                newOutput = np.zeros((nq,nt,int(np.round(ns*1.5))),dtype=output.dtype) # Make a bigger output array
                newOutput[:,:,:ns] = output # Copy data over into bigger array
                output = newOutput # Switch over to the new output array

            output[iquant,itrack-1,ioutput:ioutput+chunklen] = data[idata:idata+chunklen,iquant] # Paste the data

            idata += chunklen # Set cursor past the pasted data in the 'data' attribute
            ioutput_all[iquant, itrack-1] += chunklen # Set cursor to the next open row in the output

    return output