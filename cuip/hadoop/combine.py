#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing
import numpy as np
from datetime import datetime
from itertools import izip_longest
from scipy.ndimage import imread
from cuip.cuip.utils import cuiplogger
from cuip.cuip.database.add_files import UpdateTask
from cuip.cuip.database.db_tables import ToFilesDB
from cuip.cuip.database.database_worker import Worker
from cuip.cuip.utils.misc import get_files, _get_files

logger = cuiplogger.cuipLogger(loggername="COMBINE", tofile=False)
dbname = os.getenv("CUIP_DBNAME")
tablename = "test_uo_files" 
# -- define the function for stacking images
def merge_subset(conn, sublist, dpath, binfac, nimg_per_file, nrow=2160, 
                 ncol=4096, nwav=3, verbose=False):
    """
    Take a list of lists, merge each sublist into a stacked image, and 
    write to disk.
    """
    dr       = nrow//binfac
    dc       = ncol//binfac
    img_out = np.zeros([nimg_per_file*dr, dc, nwav], dtype=np.uint8)

    for gid, tflist in sublist.items():
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
        #newfname = os.path.join(dpath, os.path.basename(tflist[0]))[:-3]+"raw"
        newfname = os.path.join(dpath, "{0}.raw".format(gid))
        logger.info("Writing group: "+str(gid))
        img_out.tofile(newfname)
        img_out[:] = 0

    print "---"
    conn.close()
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
    en_time = "15.59.59"

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

    # -- partition the file list into output files and processors in the form of dictionary
    # -- where key = group id and value = list of files in that group
    flist_out = {i: file_list[i*nimg_per_file:(i+1)*nimg_per_file] for i in range(nout)}

    # -- set the number of processors
    nproc = 2
    logger.info("Creating %s worker processes"%(nproc))

    # -- set the number of files per processor
    if nproc == 1:
        nout_per_proc = nout 
    elif nout % nproc == 0:
        nout_per_proc = nout//nproc
    else:
        nout_per_proc = nout//nproc + 1

    # -- alert the user
    logger.info("combining {0} input files into {1} output files." \
                    .format(nin,nout))

    # ----------------------------------------
    # database related calls
    def _poison_workers(tasks):
        for i in range(num_workers):
            tasks.put(None)

    # create 1 worker
    # -- Do not increase to more than 1
    num_workers = 1
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    workers     = [ Worker(dbname, tablename, tasks, results)
                    for i in range(num_workers) ] 
    # start the workers
    for worker in workers:
        worker.start()
                
    # reset group ids in the database
    tasks.put(UpdateTask(flist=None, values_to_update={'gid':-99}))

    # set the group ids
    for k, v in flist_out.items():
        tasks.put(UpdateTask(flist=v, values_to_update={'gid':k}))

    _poison_workers(tasks)
        
    # -----------------------------------------
    
    # -- initialize workers and execute
    parents, childs, ps = [], [], []
    result = []
    groups_per_proc  = map(lambda x: [z for z in x if z is not None], 
                           list(izip_longest(*(iter(flist_out.keys()), ) *nproc)))
    for ip in range(nproc):
        ptemp, ctemp = multiprocessing.Pipe()
        parents.append(ptemp)
        childs.append(ctemp)
        
        #lo = ip * nout_per_proc
        #hi = (ip+1) * nout_per_proc
        lo = ip * len(groups_per_proc)/nproc
        hi = (ip+1) * len(groups_per_proc)/nproc
        ps.append(multiprocessing.Process(target=merge_subset, 
                                          args=(childs[ip], {k: flist_out[k] 
                                                             for file_group in groups_per_proc[lo: hi]
                                                             for k in file_group},
                                                outpath, binfac, nimg_per_file), 
                                          kwargs={"verbose":True}))

        ps[ip].start()
        childs[ip].close()

    # -- Join all processes
    dum = [ps[ip].join() for ip in range(nproc)]
