#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing
import numpy as np
from scipy.ndimage import imread

# -- define the function for stacking images
def merge_subset(conn, sublist, dpath, binfac, nimg_per_file, nrow=2160, 
                 ncol=4096, nwav=3, verbose=False):
    """
    Take a list of lists, merge each sublist into a stacked image, and 
    write to disk.
    """
    if verbose:
        print("processor working...")

    dr      = nrow//binfac
    dc      = ncol//binfac
    img_out = np.zeros([nimg_per_file*dr, dc, nwav], dtype=np.uint8)

    for tflist in sublist:
        for ii,tfile in enumerate(tflist):
            img_out[dr*ii:dr*(ii+1)] = imread(os.path.join(dpath, tfile)) \
                [::binfac, ::binfac]
        
        # img_out.tofile(os.path.join("test_out", 
        #                             ".".join([i.replace(".png","") for i in 
        #                              tflist]) + ".raw"))
        img_out.tofile(os.path.join("test_out", 
                                    tflist[0].replace(".png",".raw")))
        img_out[:] = 0

    return


# -- get the file list
dpath = "temp_images"
flist = sorted(os.listdir(dpath))
nin   = len(flist)


# -- set the number of processors
nproc = 3


# -- set the binning and determine the number of output files
binfac        = 2
nimg_per_file = 4 * binfac * binfac
nout          = nin // nimg_per_file + 1*((nin % nimg_per_file) > 0)


# -- set the number of files per processor
if nproc == 1:
    nout_per_proc = nout 
elif nout % nproc == 0:
    nout_per_proc = nout//nproc
else:
    nout_per_proc = nout//nproc + 1


# -- partition the file list into output files and processors
flist_out = [flist[i*nimg_per_file:(i+1)*nimg_per_file] for i in range(nout)]


# -- initialize workers and execute
parents, childs, ps = [], [], []

for ip in range(nproc):
    ptemp, ctemp = multiprocessing.Pipe()
    parents.append(ptemp)
    childs.append(ctemp)

    lo = ip * nout_per_proc
    hi = (ip+1) * nout_per_proc
    ps.append(multiprocessing.Process(target=merge_subset, 
                                      args=(childs[ip], flist_out[lo:hi], 
                                            dpath, binfac, nimg_per_file), 
                                      kwargs={"verbose":True}))

    ps[ip].start()

dum = [ps[ip].join() for ip in range(nproc)]
