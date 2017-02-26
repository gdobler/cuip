#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from cuip.cuip.utils.misc import get_files

class LightCurves(object):
    """
    Lightcurves from a night of observations by the CUSP Urban Observatory.
    """

    def __init__(self, st, en):
        """
        Return an object containing light curves from the start to end time.
        """

        self.st = st
        self.en = en

        return

    def get_file_list(self, night=True):
        """
        Get the full file list for all files between the start and end time.
        """

        # -- get the date range
        print("PUT DT CONVERSTION IN __INIT__!!!")
        self.st  = datetime.strptime(self.st, "%Y.%m.%d")
        self.en  = datetime.strptime(self.en, "%Y.%m.%d")
        ind      = int(sys.argv[3])

        # -- query the database
        db      = os.getenv("CUIP_DBNAME")
        self.fl = get_files(db, self.st, self.en, df=True)

        # -- pull off nighttimes
        if night:
            self.fl = self.fl[(self.fl.timestamp.dt.hour >= 19) | 
                              (self.fl.timestamp.dt.hour < 5)]

        return

    def gen_light_curves(self):
        """
        Generate light curves.
        """

        print("'gen_light_curves' METHOD NOT IMPLEMENTED YET!!!")

        # -- get the full file 

        return

    @staticmethod
    def get_registration(iden):
        """
        Check if this image is registered and return the registration
        parameters if so.
        """

        print("'check_registration' STATIC METHOD NOT IMPLEMENTED YET!!!")

        return
