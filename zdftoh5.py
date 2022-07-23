###########################################
### This script converts osiris ZDF data to HDF5 file format
### It works by reading the data and metadata from zdf and then
### creating a HDF5 file from scratch. It only reads some of the
### metadata, such as the axis properties and simulation time.
### If you need more metadata you may need to update the script.
### The new ZDF files will be in a new directory created adjacent
### to the ZDF data.
###
### To use the script, just modify the "savefolders" section below.
### You can specify any simulation sub-folders that you wish to convert
### as well as the cadence/stride. For grid data you also specify
### the degree of spatial averaging; to not use spatial averaging,
### just set the averaging to [1,1]. RAW data may also be converted.
###
### Please note that the current python ZDF library is extremely slow
### for chunked datasets. RAW files in particular can take several
### minutes to read when they are saved from large simulations. You
### can specify the max number of processes to use to convert the
### data with the max_cpus variable.
###
### To use the file, please either set the "folders" variable to a
### list of simulation root directories, or run it from the command
### line with those folders as arguments. If you run it from the
### command line you may also add a "-f" flag to force it to overwrite
### any previously-saved HDF5 files (or add "-f" to the flags list if
### not running from the command line).
###
### Author: Ryan Peterson, July 2022
###########################################

import numpy as np
from glob import glob
import zdf, h5py
import os,sys
from scipy.signal import convolve2d
import multiprocessing

#----------------------------------------------
def zdfToHdf5_grid(params):
    #print('GRID '+params[2]+' '+params[0])
    #return

    fname_in,fname_out,quant,ave = params
    if '-f' not in flags and os.path.exists(fname_out): return
    print('processing',fname_out.split('/')[-1])

    savedata,info = zdf.read(fname_in)
    dtype = savedata.dtype
    savedata = savedata.T
    nx = savedata.shape[0]

    #print('dtype',dtype)

    if ave[0]>1 and ave[1]>1:
        if ave[0]<1: ave[0]=1
        if ave[1]<1: ave[1]=1
        kernel = np.ones(ave)/np.prod(ave)
        convolved = convolve2d(savedata,kernel,mode='valid')
        savedata = convolved[::ave[0],::ave[1]]

    x0,x1 = info.grid.axis[0].__dict__['min'], info.grid.axis[0].__dict__['max']
    y0,y1 = info.grid.axis[1].__dict__['min'], info.grid.axis[1].__dict__['max']

    empty_mask = np.sum(np.abs(savedata),axis=(1,2))==0
    if empty_mask[-1]:
        inds = np.where(1-empty_mask)[0]
        if len(inds)==0: ilast = 1
        else: ilast = max(1,inds[-1])
        savedata = savedata[:ilast,:]

    x1 = x0+(x1-x0)*(savedata.shape[0]*ave[0]/nx)

    os.makedirs('/'.join(fname_out.split('/')[:-1]),exist_ok=True)
    with h5py.File(fname_out,'w') as hf:
        attrs = info.iteration.__dict__
        hf.attrs['TIME'] = np.array([attrs['t']])
        hf.attrs['TIME UNITS'] = attrs['tunits']
        hf.attrs['ITER'] = np.array([attrs['n']])
        attrs = info.grid.__dict__
        hf.attrs['NAME'] = attrs['name']
        hf.attrs['LABEL'] = attrs['label']
        hf.attrs['UNITS'] = attrs['units']

        haxis = hf.create_group('AXIS')
        for iax,(axis,x0,x1) in enumerate(zip(info.grid.axis,[x0,y0],[x1,y1])):
            attrs = axis.__dict__
            name,label,units = [axis.__dict__[e] for e in ['name','label','units']]
            # hax = haxis.create_group('AXIS'+str(iax+1))
            hax = haxis.create_dataset('AXIS'+str(iax+1),data = np.array([x0,x1]))# = attrs['min']
            hax.attrs['LONG_NAME'] = attrs['name']
            hax.attrs['NAME'] = attrs['label']
            hax.attrs['TYPE'] = attrs['type']
            hax.attrs['UNITS'] = attrs['units']

        # print([[e,hf.attrs[e]] for e in hf.attrs])
        hf.create_dataset(quant,savedata.T.shape,dtype=dtype,data=savedata.T.astype(dtype))


    return

#----------------------------------------------
def zdfToHdf5_raw(params):
    #print('RAW '+params[0])
    #return

    fname_in,fname_out = params[:2]
    if '-f' not in flags and os.path.exists(fname_out): return
    print('processing',fname_out.split('/')[-1])

    savedata,info = zdf.read(fname_in)

    os.makedirs('/'.join(fname_out.split('/')[:-1]),exist_ok=True)
    hf = h5py.File(fname_out,'w')
    attrs = info.iteration.__dict__
    hf.attrs['TIME'] = np.array([attrs['t']])
    hf.attrs['TIME UNITS'] = attrs['tunits']
    hf.attrs['ITER'] = np.array([attrs['n']])

    for key in savedata:
        keydata = savedata[key]
        hf.create_dataset(key,keydata.shape,dtype=keydata.dtype,data=keydata)
    hf.close()

    return

#----------------------------------------------
def zdfToHdf5(params):
    fname_in = params[0]
    if '/RAW/' in fname_in: zdfToHdf5_raw (params)
    else:                   zdfToHdf5_grid(params)


#----------------------------------------------
args = sys.argv[1:]
flags = ["-f"]
folders = ["/home/alexis/Science/Stanford/Simulations/test3D"]
for a in args:
    if a[0] == '-': flags.append(a)
    else: folders.append(a)

max_cpus = 1
Ncpus = min(max_cpus,multiprocessing.cpu_count())

# Specify which folders you wish to convert, the frequency of dumps to convert, and the spatial averaging
savefolders = [
    #[name,            stride, ave]
    ['MS/DENSITY/*/*/', 1,  [1,1]], # Convert every density data file to HDF5 with 8x8 spatially-averaged data
    # ['MS/DENSITY/*/*/', 20, [1,1]], # Convert one in every 20 density files to HDF5 without spatial averaging
    # ['MS/FLD/*/',       1,  [8,8]], # Convert every field data file to HDF5 with 8x8 spatially-averaged data
    # ['MS/FLD/e*/',      20, [1,1]], # Convert one in every 20 E-field files to HDF5 without spatial averaging
    # ['MS/RAW/*/',       1        ], # Convert all RAW files to hdf5
]

file_info = []
for folder in folders:
    if folder[-1] != '/': folder += '/'

    for savefolder in savefolders:
        subfolder,stride = savefolder[:2]
        if subfolder[-1] != '/': subfolder += '/'
        quantfolders = sorted(glob(folder+subfolder))
        for quantfolder in quantfolders:
            quant = quantfolder.split('/')[-2]
            if '-savg' not in quant:
                print('quant',quantfolder)
                fnames = sorted(glob(quantfolder+'*.zdf'))
                for i,fname in enumerate(fnames):
                    if 'MS/RAW/' in quantfolder: # Raw data
                        if i%stride==0:
                            fname_out = quantfolder[:-1]+'-hdf5/' + fname.split('/')[-1].replace('.zdf','.h5')
                            file_info.append([fname,fname_out])
                    else: # Grid data
                        ave = savefolder[2]
                        if i%stride==0:
                            if ((ave[0]>1) or (ave[1]>1)):
                                fname_out = quantfolder[:-1]+'-savg/' + fname.split('/')[-1].replace('.zdf','.h5').replace(quant,quant+'-savg-{:d}-{:d}'.format(ave[0],ave[1]))
                            else:
                                fname_out = quantfolder[:-1]+'/' + fname.split('/')[-1].replace('.zdf','.h5')
                            file_info.append([fname,fname_out,quant,ave])


#----------------------------------------------
with multiprocessing.Pool(Ncpus) as p:
    p.map(zdfToHdf5,file_info)
