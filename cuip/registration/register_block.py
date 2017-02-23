#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import uo_tools as ut
from datetime import datetime
from register import *
from cuip.cuip.utils.misc import get_files

if __name__=="__main__":

    # -- get time
    t0 = time.time()

    # -- get the date range
    st  = datetime.strptime(sys.argv[1], "%Y.%m.%d")
    en  = datetime.strptime(sys.argv[2], "%Y.%m.%d")
    ind = int(sys.argv[3])

    # -- query the database
    db = os.getenv("CUIP_DBNAME")
    fl = get_files(db, st, en, df=True)

    # -- pull off nighttimes
    fl  = fl[(fl.timestamp.dt.hour >= 19) | (fl.timestamp.dt.hour < 5)]
    nfl = len(fl)

    # -- open the log file
    lopen = open(os.path.join("output", "register_{0:04}.log".format(ind)), 
                              "w")
    lopen.write("Registering {0} files...\n========\n".format(nfl))

    # -- register (use default catalog)
    dr, dc, dt = [], [], []
    for ii, (rind, row) in enumerate(fl.iterrows()):
        if ii % 10 == 0:
            lopen.write("  registering file {0}\n".format(ii))
            lopen.flush()
        infile = os.path.join(row.fpath, row.fname)
        try:
            params = register(ut.read_raw(infile))
            dr.append(params[0])
            dc.append(params[1])
            dt.append(params[2])
        except:
            dr.append(-9999)
            dc.append(-9999)
            dt.append(-9999)
        if (ii + 1) % 100 == 0:
            flt           = fl[:ii+1].copy()
            flt["drow"]   = dr
            flt["dcol"]   = dc
            flt["dtheta"] = dt
            flt.to_csv(os.path.join("output", "register_{0:04}.csv" \
                                        .format(ind)), index=False)

    # -- add to dataframe
    fl["drow"]   = dr
    fl["dcol"]   = dc
    fl["dtheta"] = dt

    # -- write to csv
    lopen.write("\nWriting to csv...\n========\n")
    lopen.flush()
    fl.to_csv(os.path.join("output", "register_{0:04}.csv".format(ind)), 
              index=False)
    lopen.write("FINISHED in {0}s\n".format(time.time() - t0))
    lopen.flush()
    lopen.close()
