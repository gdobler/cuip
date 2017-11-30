from __future__ import print_function

import os
import sys
import time
import cPickle
import datetime
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as ndm
try:
    from plot import *
except:
    pass


class LightCurves(object):
    def __init__(self, lpath, vpath, rpath, spath, outp):
        """"""

        # -- Data paths.
        self.lpath = lpath
        self.vpath = vpath
        self.rpath = rpath
        self.spath = spath
        self.outp = outp

        # -- Create metadata df.
        self._metadata(self.rpath, os.path.join(self.outp, "null_data.pkl"))

        # -- Load bigoffs if they have been saved to file.
        bigo_path = os.path.join(self.outp, "bigoffs.pkl")

        if os.path.isfile(bigo_path):
            self._load_bigoffs(bigo_path)
        else:
            self._write_bigoffs(bigo_path) # Est. Time: 7 mins.lc.

        # -- Load appertures.
        self._load_wins(os.path.join(self.spath, "window_labels.out"))

        # -- Find apperture centers.
        bbl_path = os.path.join(self.spath, "12_3_14_bblgrid_clean.npy")
        self._src_centers(bbl_path)

        # -- Create bbl dicts.
        pluto_mn = os.path.join(self.spath, "pluto", "MN.csv")
        self._bbl_dicts(bbl_path, pluto_mn)


    def _metadata(self, rpath, null_path):
        """Load all registration files and concatenate to pandas df. Find
        indices for every day in corresponding on/off/lightcurve files.
        Args:
            rpath (str) - path to registration parent folder.
        """

        tstart = time.time()
        print("LIGHTCURVES: Creating metadata...                              ")
        sys.stdout.flush()

        # -- Load null values data.
        tmp = pd.DataFrame.from_dict(pd.read_pickle(null_path), orient="index")
        tmp = tmp[tmp.null_per < 0.25]
        # -- Extract null sources to be ignored.
        self.ign_src = list(set(src for l in tmp.null_scr for src in l))

        # -- Create metadata dfs from each registration file.
        dfs = []
        for rfile in sorted(os.listdir(rpath)):
            # -- Read the current registration file.
            rfilepath = os.path.join(rpath, rfile)
            cols = ["timestamp"]
            reg = pd.read_csv(rfilepath, parse_dates=cols, usecols=cols)
            # -- Subselect for times fromm 9PM - 5AM.
            hours = reg.timestamp.dt.hour
            reg = reg[(hours > 20) | (hours < 12)].reset_index()
            # -- Shift data so each continuours night shares a common date.
            tdelta = datetime.timedelta(0, 6 * 3600)
            reg.timestamp = reg.timestamp - tdelta
            # -- Find min and max idx for each day.
            grpby = reg.groupby(reg.timestamp.dt.date)["index"]
            min_idx = pd.DataFrame(grpby.min())
            max_idx = pd.DataFrame(grpby.max() + 1)
            # -- Create meta df.
            meta = min_idx.merge(max_idx, left_index=True, right_index=True) \
                .rename(columns={"index_x": "start", "index_y": "end"})
            meta["fname"] = rfile[-8:-4]
            meta["timesteps"] = meta.end - meta.start
            dfs.append(meta)

        # -- Concatenate metadata dataframe, only keeping days with < 25% null.
        self.meta = pd.concat(dfs).loc[tmp.index]

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))


    def _load_bigoffs(self, inpath):
        """Load bigoffs .pkl.
        Args:
            input (str) - filepath to bigoffs.
        """

        tstart = time.time()
        print("LIGHTCURVES: Loading bigoffs from file...                      ")
        sys.stdout.flush()
        self.bigoffs = pd.read_pickle(inpath).set_index("index")

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def _write_bigoffs(self, outpath):
        """Calculate the bigoff for all nights in the metadata and save the
        resulting dictionary to file.
        Args:
            output (str) - filepath to save bigoff.
        """

        tstart = time.time()

        self.bigoffs = {}
        ll = len(self.meta.index.unique())
        for idx, dd in enumerate(self.meta.index.unique()):
            self.loadnight(dd)

            print("LIGHTCURVES: Calculating bigoffs... ({}/{})               " \
                .format(idx + 1, ll))
            sys.stdout.flush()
            self.bigoffs[dd] = self._find_bigoffs()

        self._bigoffs_df()
        self.bigoffs.columns = self.bigoffs.columns + 1

        print("LIGHTCURVES: Writing bigoffs to file...                        ")
        sys.stdout.flush()
        self.bigoffs.reset_index().to_pickle(outpath)

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


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


    def _bigoffs_df(self):
        """Create dataframe of bigoffs."""

        tstart = time.time()
        print("LIGHTCURVES: Creating bigoffs df...                      ")
        sys.stdout.flush()

        # -- Create dataframes for each day.
        dfs = []
        for ii in self.bigoffs.keys():
            df = pd.DataFrame.from_dict(self.bigoffs[ii]) \
                .rename(columns={0: "src", 1: ii.strftime("%D"), 2: "diff"}) \
                .set_index("src").drop("diff", axis=1)
            dfs.append(df.T)

        # -- Concatenate all daily dfs.
        self.bigoffs = pd.concat(dfs)
        self.bigoffs.index = pd.to_datetime(self.bigoffs.index)
        self.bigoffs.replace(0., np.nan, inplace=True)

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def _load_wins(self, inpath):
        """Load apperture map.
        Args:
            inpath (str) - filepath to apperture map.
        """

        tstart = time.time()
        print("LIGHTCURVES: Loading appertures from file...                   ")
        sys.stdout.flush()

        nrow = 2160
        ncol = 4096
        buff = 20

        # - Load light sources.
        self.srcs = np.zeros((nrow, ncol), dtype=bool)
        self.srcs[buff:-buff, buff:-buff] = np.fromfile(inpath, int) \
            .reshape(nrow - 2 * buff, ncol - 2 * buff) \
            .astype(bool)

        # -- Create label map.
        self.labs = ndm.label(self.srcs)[0]
        # -- Replace sources to be ignored.
        shp1, shp2 = self.labs.shape
        self.labs[np.in1d(self.labs, self.ign_src).reshape(shp1, shp2)] = 0
        # -- Use replaced labels to update srcs.
        self.srcs = self.labs > 0
        # -- Re-label
        self.labs = ndm.label(self.srcs)[0]

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def _src_centers(self, bbl_path):
        """Calculated light source centers (dictionary with idx: (xx. yy).
        Args:
            bbl_path (str) - path to bbl map.
        """

        tstart = time.time()
        print("LIGHTCURVES: Calculating light source centers...               ")
        sys.stdout.flush()

        # -- List of labels to find center for.
        labrange = range(1, self.labs.max() + 1)
        # -- Find center coordinates.
        xy = ndm.center_of_mass(self.srcs, self.labs, labrange)
        # -- Create dict of labels: (xx, yy)
        self.coords = {self.labs[int(coord[0])][int(coord[1])]:
            (int(coord[1]), int(coord[0])) for coord in xy}

        # -- Find bbl for each source.
        bbls = np.load(bbl_path)
        xx, yy = zip(*self.coords.values())
        coord_bbls = [bbls[yy[ii] - 20][xx[ii] - 20] for ii in range(len(xx))]
        self.coord_bbls = dict(zip(self.coords.keys(), coord_bbls))

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def _bbl_dicts(self, bbl_path, pluto_path):
        """Create dictionaries required for plotting bbl images."""

        # -- Load bbls, and replace 0s.
        bbls = np.load(bbl_path)
        np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)
        # -- Load PLUTO.
        df = pd.read_csv(pluto_path, usecols=["BBL", "BldgClass"]) \
            .set_index("BBL")
        df = df[[isinstance(ii, str) for ii in df.BldgClass]]

        # -- Create dict of BBL:BldgClass
        self.bbl_class = {bbl: df.loc[bbl].BldgClass for bbl in
            filter(lambda x: x in df.index, np.unique(bbls))}
        # -- Create dict of BldgClass:num.
        uniq_class = np.unique(self.bbl_class.values())
        self.classn = dict(zip(uniq_class, range(len(uniq_class))))
        self.classn_r = {v: k for k, v in self.classn.items()}
        # -- Create dict of BBL:num.
        self.bbln = dict(zip(self.bbl_class.keys(),
            [self.classn[v] for v in self.bbl_class.values()]))


    def _find_err_data(self):
        """Find sources (indices) that are null and total null percentage for a
        given night."""

        # -- Find sources included in frame.
        # -- Calculate the mean over all sources.
        smean = self.lightc.mean(axis=0)
        # -- Find sources that are all null.
        outframe = smean == -9999
        if sum(outframe) > 0:
            # -- Find the idx of all such sources.
            null_src = [i for i, x in enumerate(outframe) if x == True]
        else:
            null_src = []

        self.null_src = null_src
        self.null_per = float(sum(sum(self.lightc == -9999))) / self.lightc.size


    def _loadfiles(self, fname, start, end):
        """Load lightcurves, ons, and offs.
        Args:
            fname (str) - fname suffix (e.g., '0001').
            start (int) - idx for the start of the evening.
            end (int) - idx for the end of the evening.
        Returns:
            (list) - [lightc, on, off]
        """

        # -- Assign filepaths.
        lc_fpath = os.path.join(self.lpath, "light_curves_{}.npy".format(fname))
        ons_fpath  = os.path.join(self.vpath, "good_ons_{}.npy".format(fname))
        offs_fpath = os.path.join(self.vpath, "good_offs_{}.npy".format(fname))

        # -- Load .npy files and select for the correct night.
        dimg = np.load(lc_fpath).mean(-1)[start: end]
        ons = np.load(ons_fpath)[start: end]
        offs = np.load(offs_fpath)[start: end]

        # -- Remove sources that are not in all images.
        dimg = np.delete(dimg, self.ign_src, axis=1)
        ons = np.delete(ons, self.ign_src, axis=1)
        offs = np.delete(offs, self.ign_src, axis=1)

        return [dimg, ons, offs]


    def loadnight(self, night):
        """Load a set of lightcurves, ons, and offs.
        Args:
            night (datetime.date) - night to load data for.
        """

        self.night = night
        metadata = self.meta[self.meta.index == night].to_dict("records")

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

        self._find_err_data()

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))


if __name__ == "__main__":
    # -- Load environmental variables.
    LIGH = os.environ["LIGHTCURVES"]
    VARI = os.environ["VARIABILITY"]
    REGI = os.environ["REGISTRATION"]
    SUPP = os.environ["SUPPPATH"]
    OUTP = os.environ["OUTPATH"]

    # -- Create LightCurve object.
    lc = LightCurves(LIGH, VARI, REGI, SUPP, OUTP)

    # -- Load a specific night for plotting.
    lc.loadnight(pd.datetime(2014, 7, 29).date())
