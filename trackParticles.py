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
    # Output: a dict containing the track data. Keys are ['t', 'q', 'ene', 'x1', 'x2', 'p1', 'p2', 'p3']
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


"""
#--------------------------------------------------------------
def getTrackData(path, species):
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
    # Output: a dict containing the track data. Keys are ['t', 'q', 'ene', 'x1', 'x2', 'p1', 'p2', 'p3']
    #    If you enabled ifdmp_tracks_efl or ifdmp_tracks_bfl, it includes ['E1', 'E2', 'E3', 'B1', 'B2', 'B3']
    #    Each element of the dict is a 2D array with dimensions (particle_index, timestep)

    with h5py.File(path+'/MS/TRACKS/'+species+'-tracks.h5', 'r') as tracks:
        labels = [s.decode() for s in tracks.attrs['QUANTS']] # Get quant labels as regular (non-binary) strings
        data = tracks['data'][:] # Get the data as a numeric array

        itermap = tracks['itermap'][:] # Get itermap as a numeric array
        ntracks = tracks.attrs['NTRACKS'][0]  #number of individual macroparticles tracked
        table_data = reorderTableNumba(data, itermap, ntracks)

    output_dict = {}
    for i,label in enumerate(labels[1:]):
        output_dict[label] = table_data[i,:,:]

    return output_dict


#--------------------------------------------------------------
@numba.njit() # This is a faster version of reorderTable using Numba. You can comment this line out if you don't want to use Numba.
# def reorderTableNumba(data, itermap, ntracks):
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
"""

"""
def getTrackDataByIndex(fname,indices=[]):
    # Read chunked hdf5 track data for a few tracks.
    # This works because of the 'itermap' member. Every row of itermap contains information about the next
    # several rows of the 'data' attribute. Each row of itermap is (itrack, chunklen, isimulation) where
    # itrack is the particle track id for the next chunklen rows of the 'data' member, and begins at sim step
    # isimulation. This function only loads the parts of the track file corresponding to the particles in indices.
    #
    # Input: fname: Filename of the track data
    #        indices: list of particle indices to read. If empty, all particles are returned. The indices start at 1
    #                 and correspond to the tag's position in the tag file, i.e. the first tag in your file has index=1.
    # Output: a list of dicts containing the track data. Keys are ['t', 'q', 'ene', 'x1', 'x2', 'p1', 'p2', 'p3']
    #    If you enabled ifdmp_tracks_efl or ifdmp_tracks_bfl, it may include ['E1', 'E2', 'E3', 'B1', 'B2', 'B3']
    #    Each element of the dict is a 1D array

    tracks = h5py.File(fname, 'r')
    #print([e for e in tracks.attrs])
    labels = [s.decode() for s in tracks.attrs['QUANTS']] # Get quant labels as regular (non-binary) strings
    data = tracks['data'] # Leave data as a HDF5 object since we're only going to access part of the dataset.
    itermap = tracks['itermap'][:] # Get itermap as a numeric array

    # We'll just do this without Numba since we don't want to load the whole data file. This should be faster if only a few tracks are needed.
    output_all = []
    idata_itermap = np.concatenate([[0],np.cumsum(itermap[:,1])[:-1]])  # This is the read cursor location for each chunk of data corresponding to a row of itermap
    for itrack in indices:
        output = []
        itermap_indices = np.where(itermap[:,0] == itrack)[0]
        if len(itermap_indices)==0:
            print('No track data avilable for track index '+str(itrack))
            return
        for i_itermap in itermap_indices: # for each chunk of itermap (and data) corresponding to this particle
            itrack, chunklen, isimulation = itermap[i_itermap,:] # info about this chunk of data
            idata = idata_itermap[i_itermap] # Read cursor location in data
            output.append(data[idata:idata+chunklen,:])
        output = np.concatenate(output,axis=0)
        output_dict = {}
        for i,label in enumerate(labels[1:]):
            output_dict[label] = output[:,i]
        output_all.append(output_dict)

    return output_all



a = getTrackDataByIndex("/home/alexis/Science/Stanford/Simulations/CS3Dtrack/MS/TRACKS/eL-tracks.h5",[1])
"""
