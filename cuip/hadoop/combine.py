__author__ = "Mohit Sharma"

from datetime import datetime, timedelta
from cuip.cuip.utils import cuiplogger
from cuip.cuip.utils.misc import getFiles
from itertools import islice
import numpy as np
import psycopg2
from scipy.ndimage import imread 
import multiprocessing
import os

# Logger
logger = cuiplogger.cuipLogger(loggername="BATCH", tofile=False)

# File Paths
INPATH = '/projects/cusp/10101/0/'
OUTPATH = '/home/cusp/mohitsharma44/uo/cuip/cuip/hadoop/output/bad_combined'

# Range of dates
start_date = "2013.11.17"
start_time = "17.00.00"
end_date   = "2013.11.17"
end_time   = "23.59.59"

s_year, s_month, s_day = start_date.split('.')
s_hour, s_min, s_sec   = start_time.split('.')
e_year, e_month, e_day = end_date.split('.')
e_hour, e_min, e_sec   = end_time.split('.')

start = datetime(int(s_year), int(s_month), int(s_day), int(s_hour), int(s_min), int(s_sec))
end  = datetime(int(e_year), int(e_month), int(e_day), int(e_hour), int(e_min), int(e_sec))

# Increment time by XX hours (max = 10)
incr = 1

# target gf_gen for multiprocessing

# List of file generators
files_gen_list = []
counter = 0

while True:
    #counter += 1
    start_next = start+timedelta(hours=incr)
    if start_next <= end:
        # range is between current time interation and +incr hours
        files_gen_list.append(
            getFiles(INPATH, 
                     start.strftime("%Y.%m.%d"), start.strftime("%H.%M.%S"), 
                     start_next.strftime("%Y.%m.%d"), start_next.strftime("%H.%M.%S"))
            )
        start += timedelta(hours=incr)
    else:
        # range is between current time iteration and end
        files_gen_list.append(
            getFiles(INPATH, 
                     start.strftime("%Y.%m.%d"), start.strftime("%H.%M.%S"), 
                     end.strftime("%Y.%m.%d"), end.strftime("%H.%M.%S"))
            )
        break

def groupFiles(n, gf):
    """
    Return path for `n` files in a list
    Parameters
    ----------
    n: int
        number of files to return in a list
    gf: `generator`
        generator containing all the file paths

    Returns
    -------
    flist: list
        list of n file paths
    """
    flist =  []
    for i in range(n):
        flist.append(gf.next())
    return flist

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
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
            #logger.info('%s: %s' % (proc_name, next_task))
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class StackImages(object):
    def __init__(self, img_arr=None, filelist=None, bin=1, nrows=None, ncols=None, ndims=None, outpath=None):
        self.img_arr  = img_arr
        self.filelist = filelist
        self.bin      = bin
        self.nrows    = nrows
        self.ncols    = ncols
        self.ndims    = ndims
        self.outpath  = outpath

    def __call__(self):
        self.combinedtoraw(self.img_arr, self.filelist, self.bin, self.nrows, self.ncols, self.ndims, self.outpath)
        #self.stackingProcess(self.filelist)

    def combinedtoraw(self, img_arr=None, filelist=None, bin=1, nrows=None, ncols=None, ndims=None, outpath=None):
        """
        Convert raw/ png images to raw files
        Parameters
        ----------
        img_arr: np.array
            numpy array of the size of the image
            example:
        img_arr = np.zeros([nrows*n, ncols, ndims], np.uint8)
            where `n` is the total number of images to be combined
            by stacking vertically
        filelist: list
            list containing files to be stacked together
        bin: int
            binning factor. default is 1
        nrows: int
            number of rows in a single image
        ncols: int
            number of cols in a single image
        ndims: int
            number of dimensions (RGB = 3)
        outpath: str
            path where the file should be written to
        """
        # for ind, imgs in enumerate(groupFiles(n, gfgen)):
        for ind, imgs in enumerate(filelist):
            if imgs[-3:].lower() == 'png':
                img_arr[(nrows/bin)*ind: (nrows/bin)*(ind+1), :, :] = imread(imgs, mode='RGB')[::bin, ::bin]
            elif imgs[-3:].lower() == 'raw':
                img_arr[(nrows/bin)*ind: (nrows/bin)*(ind+1), :, :] = np.fromfile(imgs, dtype=np.uint8).reshape(nrows, ncols, ndims)[::bin, ::bin]
            else:
                logger.error("File Format not supported "+str(imgs))
        # Rename the extension of the file to .raw.
        # outfile is mtime of the file
        # newfname = os.path.basename(imgs)[:-4]+".raw"
        newfname = str(os.path.getmtime(imgs))+".raw"
        logger.info("Writing: "+str(newfname))
        img_arr.tofile(os.path.join(outpath, newfname))

    def stackingProcess(self, filelist, **kwargs):
        pass
        #for i in filelist:
        #    logger.info(i)


if __name__ == "__main__":
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    
    # Files to be comined together
    combine = 64
    BIN = 8

    # File shapes
    nrows = 2160
    ncols = 4096
    ndims = 3

    # Empty Array
    comb_imgs = np.zeros([nrows*combine/BIN, ncols/BIN, ndims], np.uint8)
    
    # Start consumers
    num_workers = 20#multiprocessing.cpu_count()/ 4
    logger.info('Creating %d Workers' % num_workers)
    workers = [ Worker(tasks, results)
                for i in xrange(num_workers) ]
    for w in workers:
        w.start()

    def poisonChildren():
        # Add a poison pill for each consumer
        for i in xrange(num_workers):
            tasks.put(None)
    
    # Enqueue jobs
    filectr = 0
    num_jobs = len(files_gen_list)
    try:
        for gens in files_gen_list:
            while True:
                filelist = [z for z in islice(gens, combine)]
                filectr += len(filelist)
                if filelist:
                    tasks.put(StackImages(comb_imgs, filelist, BIN, nrows, ncols, ndims, OUTPATH))
                else:
                    logger.info(filectr)
                    logger.info("Tasks Added")
                    break
    finally:
        poisonChildren()
        # Wait for all of the tasks to finish
        tasks.join()
    
    # Start printing results
    #while num_jobs:
    #    result = results.get()
    #    print 'Result:', result
    #    num_jobs -= 1
