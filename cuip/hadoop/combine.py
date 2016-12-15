#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing
import numpy as np
from datetime import datetime
from scipy.ndimage import imread
from cuip.cuip.utils import cuiplogger
from cuip.cuip.utils.misc import get_files, _get_files

logger = cuiplogger.cuipLogger(loggername="COMBINE", tofile=False)

# -- define the function for stacking images
def merge_subset(conn, sublist, dpath, binfac, nimg_per_file, nrow=2160, 
                 ncol=4096, nwav=3, verbose=False):
    """
    Take a list of lists, merge each sublist into a stacked image, and 
    write to disk.
    """
    dr      = nrow//binfac
    dc      = ncol//binfac
    img_out = np.zeros([nimg_per_file*dr, dc, nwav], dtype=np.uint8)

    for tflist in sublist:
        for ii,tfile in enumerate(tflist):
            ext = tfile[-3:].lower()
            if ext == 'png':
                img_out[dr*ii:dr*(ii+1)] = imread(tfile, mode="RGB") \
                    [::binfac, ::binfac]
            elif ext == 'raw':
                img_out[dr*ii:dr*(ii+1)] = np.fromfile(tfile, dtype=np.uint8) \
                    .reshape(nrow, ncol, nwav)[::binfac, ::binfac]
            else:
                logger.error("File format not supported "+str(tfile))

        newfname = os.path.join(dpath, os.path.basename(tflist[0]))[:-3]+"raw"

        logger.info("Writing: "+newfname)
        img_out.tofile(newfname)
        img_out[:] = 0

    return

if __name__ == "__main__":

    # -- get the file list
    inpath  = os.getenv("CUIP_2013")
    outpath = "output/combined_images"
    dbname  = os.getenv("CUIP_DBNAME")
        
    # set start and end times
    st_date = "2013.11.17"
    st_time = "15.00.00"
    en_date = "2013.11.17"
    en_time = "23.59.59"

    st = datetime(*[int(i) for i in st_date.split(".") + st_time.split(".")])
    en = datetime(*[int(i) for i in en_date.split(".") + en_time.split(".")])

    # -- get all the files between st and en
    if dbname:
        logger.info("Fetching file locations from database")
        file_list = get_files(dbname, st, en)
    else:
        logger.warning("Database not found. Process continue by scanning filesystem")
        logger.warning("This might take longer")
        # get files by scanning the file system 
        file_list = []
        file_gen_list = _get_files(inpath, st, en)
        for all_files in file_gen_list:
            file_list.append(all_files)

    nin = len(file_list)

    # -- set the binning and determine the number of output files
    binfac        = 2
    nimg_per_file = 4 * binfac * binfac
    nout          = nin // nimg_per_file + 1*((nin % nimg_per_file) > 0)

    # -- partition the file list into output files and processors
    flist_out = [file_list[i*nimg_per_file:(i+1)*nimg_per_file] for i in range(nout)]

    # -- set the number of processors
    nproc = 16
    logger.info("Creating %s worker processes"%(nproc))

    # -- set the number of files per processor
    if nproc == 1:
        nout_per_proc = nout 
    elif nout % nproc == 0:
        nout_per_proc = nout//nproc
    else:
        nout_per_proc = nout//nproc + 1

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
                                                outpath, binfac, nimg_per_file), 
                                          kwargs={"verbose":True}))
        
        ps[ip].start()

    # -- Join all processes
    dum = [ps[ip].join() for ip in range(nproc)]
