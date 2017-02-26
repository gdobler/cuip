#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing
import numpy as np
from datetime import datetime
from itertools import izip_longest
from scipy.ndimage import imread
from cuip.cuip.utils import cuiplogger
from cuip.cuip.database.add_files import UpdateTask, ToCSV
from cuip.cuip.database.db_tables import ToFilesDB
from cuip.cuip.database.database_worker import Worker
from cuip.cuip.utils.misc import get_files, _get_files

logger = cuiplogger.cuipLogger(loggername="COMBINE", tofile=False)

# -- define the function for stacking images
def merge_subset(conn, sublist, dpath, binfac, nimg_per_file, nrow=2160, 
                 ncol=4096, nwav=3, verbose=False):
    """
    Take a list of lists, merge each sublist into a stacked image, and 
    write to disk.
    """
    dr          = nrow//binfac
    dc          = ncol//binfac
    img_out     = np.zeros([nimg_per_file*dr, dc, nwav], dtype=np.uint8)
    for gid, tflist in sublist.items():
        filenumbers = []
        for ii,tfile in enumerate(tflist):
            filenumbers.append(str(tfile[1]))
            ext = tfile[0][-3:].lower()
            if ext == 'png':
                img_out[dr*ii:dr*(ii+1)] = imread(tfile[0], mode="RGB") \
                    [::binfac, ::binfac]
            elif ext == 'raw':
                img_out[dr*ii:dr*(ii+1)] = np.fromfile(tfile[0], dtype=np.uint8) \
                    .reshape(nrow, ncol, nwav)[::binfac, ::binfac, ::-1]
            else:
                logger.error("File format not supported "+\
                                 str(tfile[0]) + "fnumber: "+\
                                 str(tfile[1]))
        #newfname = os.path.join(dpath, "{0}.raw".format(gid))
        newfname = os.path.join(dpath, "_".join([filenumbers[0], 
                                                 filenumbers[-1]])+str(".raw"))
        logger.info("Writing filename: "+str(newfname))
        img_out.tofile(newfname)
        img_out[:] = 0
    conn.close()
    return 

if __name__ == "__main__":

    inpath  = os.getenv("CUIP_2013")
    f_dbname = os.getenv("CUIP_DBNAME")
    f_tbname = os.getenv("CUIP_TBNAME")
    outpath = "output/combined_images"
        
    # set start and end times
    st_date = "2013.11.17"
    st_time = "17.00.00"
    en_date = "2013.11.17"
    en_time = "23.59.59"

    st = datetime(*[int(i) for i in st_date.split(".") + st_time.split(".")])
    en = datetime(*[int(i) for i in en_date.split(".") + en_time.split(".")])

    # -- get all the files between st and en
    if f_dbname:
        logger.info("Fetching file locations from database. This will return fname, fpath and fnumber")
        file_list = get_files(f_dbname, st, en)
    else:
        logger.warning("Database not found. Process continue by scanning filesystem")
        logger.warning("This might take longer and will only return fname and fpath")
        # get files by scanning the file system 
        file_list = []
        file_gen_list = _get_files(inpath, st, en)
        for all_files in file_gen_list:
            file_list.append(all_files)

    nin = len(file_list)

    # -- set the binning and determine the number of output files
    binfac        = 8
    nimg_per_file = 4 * binfac * binfac
    nout          = nin // nimg_per_file + 1*((nin % nimg_per_file) > 0)

    # -- partition the file list into output files and processors 
    # -- in the form of dictionary
    # -- where key = group id and value = list of files in that group
    flist_out = {i: file_list[i*nimg_per_file:(i+1)*nimg_per_file] \
                     for i in range(nout)}

    # -- set the number of processors
    nproc = 10
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

    # -- database related calls
    def _poison_workers(tasks):
        """
        convinient method to kill all database workers
        """
        for i in range(num_workers):
            tasks.put(None)

    # create 1 worker
    # -- Do not increase to more than 1
    num_workers = 1
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    workers     = [ Worker(f_dbname, f_tbname, tasks, results)
                    for i in range(num_workers) ] 
    # start the workers
    for worker in workers:
        worker.start()
                
    # reset group ids in the database
    tasks.put(UpdateTask(flist=None, values_to_update={'gid':-99}))

    # set the group ids
    for k, v in flist_out.items():
        tasks.put(UpdateTask(flist=[_file[0] for _file in v], values_to_update={'gid':k}))

    # export the table filtered by modified values
    tasks.put(ToCSV(where_clause='gid', compare_value=range(len(flist_out.keys()))))

    _poison_workers(tasks)

    # -- initialize workers and execute
    parents, childs, ps = [], [], []
    result = []
    groups_per_proc = [flist_out.keys()[i::nproc] for i in range(nproc)]
    logger.info("Starting stacking process")
    for ip in range(nproc):
        ptemp, ctemp = multiprocessing.Pipe()
        parents.append(ptemp)
        childs.append(ctemp)
        ps.append(multiprocessing.Process(target=merge_subset, 
                                          args=(childs[ip], {gid: flist_out[gid] 
                                                             for gid \
                                                                 in groups_per_proc[ip]},
                                                outpath, binfac, nimg_per_file), 
                                          kwargs={"verbose":True}))

        ps[ip].start()
        childs[ip].close()

    # -- Join all processes
    dum = [ps[ip].join() for ip in range(nproc)]
