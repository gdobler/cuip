#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mohit Sharma"

import os
import multiprocessing
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from itertools import islice
from scipy.ndimage import imread 
from cuip.cuip.utils import cuiplogger
from cuip.cuip.utils.misc import get_files, _get_files

# initialize logger
logger = cuiplogger.cuipLogger(loggername="BATCH", tofile=False)


class Worker(multiprocessing.Process):
    """
    Takes the next task from task pool/queue and executes it.
    """

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue   = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                logger.info('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class StackImages(object):
    """
    Context around a function that stacks images.
    """

    def __init__(self, img_arr=None, filelist=None, fac=1, nrow=None, 
                 ncol=None, nwav=None, outpath=None):
        self.img_arr  = img_arr
        self.filelist = filelist
        self.fac      = fac
        self.nrow     = nrow
        self.ncol     = ncol
        self.nwav     = nwav
        self.outpath  = outpath

    def __call__(self):
        self.combine_images(self.img_arr, self.filelist, self.fac, self.nrow, 
                            self.ncol, self.nwav, self.outpath)

    def combine_images(self, img_arr=None, filelist=None, fac=1, nrow=None, 
                       ncol=None, nwav=None, outpath=None):
        """
        Convert raw/ png images to raw files
        Parameters
        ----------
        img_arr: np.array
            numpy array of the size of the image
            example:
        img_arr = np.zeros([nrow*n, ncol, nwav], np.uint8)
            where `n` is the total number of images to be combined
            by stacking vertically
        filelist: list
            list containing files to be stacked together
        fac: int
            binning factor. default is 1
        nrow: int
            number of rows in a single image
        ncol: int
            number of cols in a single image
        nwav: int
            number of dimensions (RGB = 3)
        outpath: str
            path where the file should be written to
        """

        for ind, tfile in enumerate(filelist):
            lo  = (nrow/fac)*ind
            hi  = (nrow/fac)*(ind+1)
            ext = tfile[-3:].lower()

            if ext == 'png':
                img_arr[lo:hi] = imread(tfile, mode='RGB')[::fac, ::fac]
            elif ext == 'raw':
                img_arr[lo:hi] = np.fromfile(tfile, dtype=np.uint8) \
                    .reshape(nrow, ncol, nwav)[::fac, ::fac]
            else:
                logger.error("File Format not supported "+str(tfile))

        # rename the extension of the file to .raw
        newfname = str(os.path.getmtime(tfile))+".raw"

        # write to file
        logger.info("Writing: "+str(newfname))
        img_arr.tofile(os.path.join(outpath, newfname))


if __name__ == "__main__":
    # setting up threadsafe tasks and results queue
    tasks   = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # set input and output paths
    inpath  = os.getenv("CUIP_2013")
    outpath = "output/combined_images"
    dbname  = os.getenv("CUIP_DBNAME")

    # set start and end times
    st_date = "2013.11.17"
    st_time = "17.00.00"
    en_date = "2013.11.17"
    en_time = "17.59.59"

    st = datetime(*[int(i) for i in st_date.split(".") + st_time.split(".")])
    en = datetime(*[int(i) for i in en_date.split(".") + en_time.split(".")])
    
    # Files to be comined together
    combine = 64
    fac     = 4

    # File shapes
    nrow = 2160
    ncol = 4096
    nwav = 3

    # Empty Array
    comb_imgs = np.zeros([nrow*combine/fac, ncol/fac, nwav], np.uint8)
    
    # Start workers
    num_workers = 20
    logger.info('Creating %d Workers' % num_workers)
    workers = [Worker(tasks, results) for i in range(num_workers)]

    for worker in workers:
        worker.start()

    def poisonChildren():
        # Add a poison pill for each worker
        for i in range(num_workers):
            tasks.put(None)
            
    # put jobs (tasks) in queue
    try:
        file_list = []
        # first try to get files from database
        if dbname:
            logger.info("Fetching file locations from database")
            file_list = get_files(dbname, st, en)
        else:
            logger.warning("Database not found. Process continue by scanning filesystem")
            logger.warning("This might take longer")
            # get files by scanning the file system
            file_gen_list = _get_files(inpath, st, en)
            for all_files in file_gen_list:
                file_list.append(all_files)
        # group of 'combine' files
        files = zip(*(iter(file_list),) * combine)
        for n_files in files:
            tasks.put(StackImages(comb_imgs, n_files, 
                                  fac, nrow, ncol, nwav,
                                  outpath))
        logger.info("All Tasks Added")
    finally:
        poisonChildren()
        # Wait for all of the tasks to finish
        tasks.join()
