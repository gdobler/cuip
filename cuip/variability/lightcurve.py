from __future__ import print_function

import os
import sys
import time
import cPickle
import datetime
import numpy as np
import pandas as pd
try:
    from plot import *
except:
    pass

class LightCurves(object):
    def __init__(self, lpath, vpath, rpath):
        """"""

        self.lpath = lpath
        self.vpath = vpath
        self.rpath = rpath
        self.meta = self.metadata(self.rpath)


    def metadata(self, rpath):
        """Load all registration files and concatenate to pandas df. Find
        indices for every day in corresponding on/off/lightcurve files.
        Args:
            rpath (str) - path to registration parent folder.
        Returns:
            df - pandas df.
        """

        tstart = time.time()

        # -- Columns to parse from csv.
        cols = ["fnumber", "timestamp"]

        print("LIGHTCURVES: Loading registration files...                     ")
        sys.stdout.flush()

        # -- Concatenated all registration csv files.
        reg = pd.concat(
            # -- Use a list comprehension to load csvs.
            [pd.read_csv(os.path.join(self.rpath, fname),
            # -- Parse dates and create a column with the fname.
            usecols=cols, parse_dates=["timestamp"]).assign(fname=fname[9:13])
            # -- List all registration files.
            for fname in sorted(os.listdir(self.rpath))]
        ).reset_index(drop=True)

        print("LIGHTCURVES: Creating metadata...                              ")
        sys.stdout.flush()
        # -- Create metadata df.
        # -- Shift data so each continuous night shares a common date.
        tdelta = datetime.timedelta(0, 6 * 3600)

        df = pd.DataFrame(
            # -- Find N records that below to each shifted day.
            reg.groupby([reg.fname, (reg.timestamp - tdelta).dt.date]).size() \
            # -- Take the cumsum for each registration file (i.e., end idx).
            .groupby(level=[0]).cumsum(), columns=["end"]
        )

        # -- Use previous end idx as start, and fill 0 for new files.
        df = df.assign(start=df.end.shift(1).fillna(0).astype(int).where(
                df.end.shift(1).fillna(0).astype(int) < df.end, 0)) \
                .reset_index()

        # -- Calculated N of timesteps for each day.
        df["timesteps"] = df["end"] - df["start"]

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))

        return df


    def _loadfiles(self, fname, start, end):
        """Load lightcurves, ons, and offs.
        Args:
            fname (str) - fname suffix (e.g., '0001').
            start (int) - idx for the start of the evening.
            end (int) - idx for the end of the evening.
        Return:
            files (list) - loaded files [lightcurve, on, off].
        """

        lc_fname = "light_curves_{}.npy".format(fname)
        ons_fname  = "good_ons_{}.npy".format(fname)
        offs_fname = "good_offs_{}.npy".format(fname)

        dimg = np.load(os.path.join(self.lpath, lc_fname)).mean(-1)[start: end]
        ons = np.load(os.path.join(self.vpath, ons_fname))[start: end]
        offs = np.load(os.path.join(self.vpath, offs_fname))[start: end]

        files = [dimg, ons, offs]

        return files


    def loadnight(self, night):
        """Load a set of lightcurve, ons, and offs.
        Args:
            night (datetime.date) - night to load data for.
        """

        self.night = night
        metadata = self.meta[self.meta.timestamp == night].to_dict("records")

        if len(metadata) == 0:
            raise ValueError("{} is not a valid night".format(night))

        tstart = time.time()
        print("LIGHTCURVES: Loading data for {}...                          " \
            .format(night))
        sys.stdout.flush()

        if len(metadata) == 1:
            fname = metadata[0]["fname"]
            start = metadata[0]["start"]
            end = metadata[0]["end"]
            self.lightc, self.on, self.off = self._loadfiles(fname, start, end)

        if len(metadata) == 2:
            data = []
            for idx in range(2):
                fname = metadata[idx]["fname"]
                start = metadata[idx]["start"]
                end = metadata[idx]["end"]
                data.append(self._loadfiles(fname, start, end))

            data = zip(data[0], data[1])
            self.lightc = np.concatenate(data[0], axis=0)
            self.on = np.concatenate(data[1], axis=0)
            self.off = np.concatenate(data[2], axis=0)

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))


    def _find_bigoffs(self):
        """Find all bigoffs for the loaded night.
        Returns:
            array - list of tuples including: (src idx, off idx, mean diff.).
        """

        # -- Get data for the given night.
        dimg = self.lightc.T
        dimg[dimg == -9999.] = np.nan
        offs = self.off.T

        # -- Loop through all sources (i.e., light curves) and find bigoff.
        bigoffs = []
        for src in range(dimg.shape[0]):
            bigoff = (src, 0, -9999)
            for off in offs[src].nonzero()[0]:
                mm = np.nanmean(dimg[src][:off]) - np.nanmean(dimg[src][off:])
                if mm > bigoff[2]:
                    bigoff = (src, off, mm)
            bigoffs.append(bigoff)

        return bigoffs


    def write_bigoffs(self, outpath):
        """Calculate the bigoff for all nights in the metadata and save the
        resulting dictionary to file.
        Args:
            output (str) - filepath to save dictionary.
        """

        tstart = time.time()

        self.bigoffs = {}
        ll = len(self.meta.timestamp.unique())
        for idx, dd in enumerate(self.meta.timestamp.unique()):
            self.loadnight(dd)

            print("LIGHTCURVES: Calculating bigoffs... ({}/{})               " \
                .format(idx + 1, ll))
            sys.stdout.flush()
            self.bigoffs[dd] = self._find_bigoffs()

        print("LIGHTCURVES: Writing bigoffs to file...                        ")
        sys.stdout.flush()
        with open(outpath, "wb") as outfile:
            cPickle.dump(self.bigoffs, outfile)

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def load_bigoffs(self, inpath):
        """Load bigoffs .pkl.
        Args:
            input (str) - filepath to bigoffs dictionary.
        """
        tstart = time.time()
        print("LIGHTCURVES: Loading bigoffs from file...                      ")
        sys.stdout.flush()
        with open(inpath, "rb") as infile:
            self.bigoffs = cPickle.load(infile)

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def bigoffs_df(self):
        """"""
        tstart = time.time()
        print("LIGHTCURVES: Creating bigoffs df...                      ")
        sys.stdout.flush()

        # -- Create dataframes for each day.
        dfs = []
        for ii in lc.bigoffs.keys():
            df = pd.DataFrame.from_dict(lc.bigoffs[ii]) \
                .rename(columns={0: "src", 1: ii.strftime("%D"), 2: "diff"}) \
                .set_index("src").drop("diff", axis=1)
            dfs.append(df.T)

        # -- Concatenate all daily dfs.
        self.df = pd.concat(dfs)
        self.df.index = pd.to_datetime(self.df.index)
        self.df.replace(0., np.nan, inplace=True)

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


if __name__ == "__main__":
    # -- Load environmental variables.
    LIGHTCURVES = os.environ["LIGHTCURVES"]
    VARIABILITY = os.environ["VARIABILITY"]
    REGISTRATION = os.environ["REGISTRATION"]
    BIGOFFS = os.environ["BIGOFFS"]

    # -- Create LightCurve object.
    lc = LightCurves(LIGHTCURVES, VARIABILITY, REGISTRATION)

    # -- Load bigoffs if they have been saved to file.
    if os.path.isfile(BIGOFFS):
        lc.load_bigoffs(BIGOFFS)
    else:
        lc.write_bigoffs(BIGOFFS) # Est. Time: 7 mins.

    # -- Create a dataframe of all bigoff times.
    lc.bigoffs_df()

    # -- Load a specific night for plotting.
    lc.loadnight(pd.datetime(2013, 11, 4).date())
    plot_lightcurve(lc, 136)
    plot_night(lc)
    plot_winter_summer_bigoffs_boxplot(lc)
