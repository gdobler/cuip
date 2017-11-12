#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script will compress the raw Audubon files to PNG.

import os
import sys
import time
import datetime
import glob
import numpy as np
import pandas as pd
from scipy.misc import imsave
from multiprocessing import Pool


def get_file_times(times_file):
    """
    Get the file times.
    """

    # -- get the file list
    dpath = os.path.join(os.environ["AUDUBON_DATA"])
    flist = []
    for root, dirs, files in os.walk(dpath):
        if "." not in root:
            print("\rcrawling {0}".format(root)),
            sys.stdout.flush()
        if root.endswith("night"):
            for tfile in files:
                if tfile.endswith(".raw"):
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


def get_file_times_fast(times_file):
    """
    Get the file times.
    """

    # -- get the file list
    dpath = os.path.join(os.environ["AUDUBON_DATA"])
    flist = sorted(glob.glob(os.path.join(dpath, "*_night", "*.raw")))
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


def compress_sub(params):
    """
    Compress a subset of the data.
    """

    # -- unpack parameters
    fnames_sub = params[0][params[2]:params[3]]
    secs_sub   = params[1][params[2]:params[3]]
    pnum       = params[4]
    sflag      = 0

    # -- image utilities
    sh = (3840, 5120)
    
    # -- open an error log
    lopen = open(os.path.join("output", 
                              "compress_audubon_file_2_proc{0:02}.log" \
                                  .format(pnum)), "w")

    for fname, ftime in zip(fnames_sub, secs_sub):
        lopen.flush()
        try:
            ofile = fname[:-3] + "png"
            imsave(ofile, np.fromfile(fname, np.uint8).reshape(sh))
        except:
            lopen.write("{0} failed to convert to {1}\n".format(fname, ofile))
            sflag = 1
            continue
        try:
            os.utime(ofile, (ftime, ftime))
        except:
            lopen.write("{0} failed to set timestamp\n".format(fname))
            sflag = 1
            continue
        try:
            pngsz = os.path.getsize(ofile)
        except:
            lopen.write("{0} failed to get filesize\n".format(fname))
            sflag = 1
            continue
        try:
            os.remove(fname)
        except:
            lopen.write("{0} failed to remove file\n".format(fname))
            sflag = 1

    # -- close the error log
    lopen.close()

    return sflag


def compress_files(times_file):
    """
    Compress the data to PNG and remove RAW file.
    """

    # -- get the filenames and times
    print("reading file times...")
    times  = pd.read_csv(times_file)
    fnames = times.filename.values
    secs   = times.time.values
    
    # -- loop through filenames and compress
    nimg  = len(times)
    nproc = 8
    dr    = int(float(nimg) / nproc + 0.5)
    t0    = time.time()
    plist = [(fnames, secs, i * dr, (i + 1) * dr, i) for i in range(nproc)]
    tpool = Pool()
    res   = tpool.map(compress_sub, plist)
    dt    = time.time() - t0
    print("ellapsed time  {0}s".format(dt))
    print("time per image {0}s".format(dt / nimg))

    return


if __name__=="__main__":

    # -- get the file times
    times_file = "output/cuip_audubon_file_times_2.csv"
    if not os.path.isfile(times_file):
        print("getting file times...")
        get_file_times_fast(times_file)

    # -- move files
    compress_files(times_file)
