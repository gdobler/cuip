#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script will compress the raw UO vis files from 2015 to PNG.

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imsave
from multiprocessing import Pool


def get_file_times(times_file):
    """
    Get the file times for the files that Mohit moved to the 10TB partition
    and write to a file (just in case...).
    """

    # -- get the file list
    dpath = os.path.join(os.environ["CUIP_2015"], "2015")
    flist = []
    for root, dirs, files in os.walk(dpath):
        if "." not in root:
            print("\rcrawling {0}".format(root)),
            sys.stdout.flush()
        for tfile in files:
            flist.append(os.path.join(dpath, root, tfile))
    nfile = len(flist)
    print("got full list of {0} files".format(nfile))


    # -- get the times and write to file
    fopen = open(times_file, "w")
    fopen.write("filename,time,year,month,day,hour,minutes,seconds\n")
    for ii, tfile in enumerate(flist):
        if (ii + 1) % 10000 == 0:
            print("\rgetting times for file {0} of {1}".format(ii + 1, nfile)),
            sys.stdout.flush()
        mtime = os.path.getmtime(tfile)
        dt    = datetime.datetime.fromtimestamp(mtime)
        fopen.write("{0},{1},{2},{3},{4},{5},{6},{7}\n" \
                        .format(tfile, mtime, dt.year, dt.month, dt.day,
                                dt.hour, dt.minute, dt.second))
    fopen.close()
    print("wrote filenames and times to file {0}".format(times_file))

    return


# # -- compress a subset of the data
# def compress_sub(params):

#     # unpack parameters
#     fnames_sub = params[0][params[3]:params[4]]
#     paths_sub  = params[1][params[3]:params[4]]
#     secs_sub   = params[2][params[3]:params[4]]
#     pnum       = params[5]
#     sflag      = 0

#     # image utilities
#     sh = (2160, 4096, 3)
    
#     # open an error log
#     lopen = open("../output/compress_file_proc{0:02}.log".format(pnum), "w")

#     for fname, fpath, ftime in zip(fnames_sub, paths_sub, secs_sub):
#         lopen.flush()
#         try:
#             ofile = os.path.join(fpath, fname.split("/")[-1][:-4] + ".png")
#             imsave(ofile, 
#                    np.fromfile(fname, np.uint8).reshape(sh)[:,:,::-1])
#         except:
#             lopen.write("{0} failed to convert to {1}\n".format(fname,ofile))
#             sflag = 1
#             continue
#         try:
#             os.utime(ofile, (ftime, ftime))
#         except:
#             lopen.write("{0} failed to set timestamp\n".format(fname))
#             sflag = 1
#             continue
#         try:
#             pngsz = os.path.getsize(ofile)
#         except:
#             lopen.write("{0} failed to get filesize\n".format(fname))
#             sflag = 1
#             continue
#         try:
#             os.remove(fname)
#         except:
#             lopen.write("{0} failed to remove file\n".format(fname))
#             sflag = 1

#     # close the error log
#     lopen.close()

#     return sflag


# def compress_files():
#     """
#     Compress the data to PNG and move to directory tree.
#     """

#     # -- set the path
#     dpath = "../data00"
    
#     # -- get the times
#     print("reading file times...")
#     times = pd.read_csv("../output/mohit_file_times.csv")
#     ntime = len(times)
#     yrs   = times.year.values
#     mos   = times.month.values
#     dys   = times.day.values
#     hrs   = times.hour.values
#     mns   = times.minutes.values
#     secs  = times.time.values
    
#     # -- loop through times and create a directory if need be 
#     print("generating paths...")
#     paths = np.array([os.path.join(dpath,str(yrs[ii]),
#                                    "{0:02}".format(int(mos[ii])),
#                                    "{0:02}".format(int(dys[ii])),
#                                    "{0:02}.{1:02}.00"\
#                                        .format(int(hrs[ii]),
#                                                int(mns[ii])//5*5)) 
#                       for ii in range(ntime)])
    
#     # -- set the filenames
#     fnames = times.filename.values
    
#     # -- loop through filenames and compress
#     nimg  = len(times)
#     nproc = 8
#     dr    = int(float(nimg)/nproc + 0.5)
#     t0    = time.time()
#     plist = [(fnames, paths, secs, i*dr, (i+1)*dr, i) for i in range(nproc)]
#     tpool = Pool()
#     res   = tpool.map(compress_sub, plist)

#     dt = time.time() - t0
#     print("ellapsed time  {0}s".format(dt))
#     print("time per image {0}s".format(dt/nimg))

#     return


if __name__=="__main__":

    # -- get the file times
    times_file = "output/cuip_2015_file_times.csv"
    if not os.path.isfile(times_file):
        print("getting file times...")
        get_file_times(times_file)

    # -- move files
    # compress_files()
