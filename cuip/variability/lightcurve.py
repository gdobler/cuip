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
from collections import defaultdict


class LightCurves(object):
    def __init__(self, lpath, vpath, rpath, spath, opath):
        """"""

        # -- Data paths.
        self.path_light = lpath
        self.path_varia = vpath
        self.path_regis = rpath
        self.path_suppl = spath
        self.path_outpu = opath

        # -- Create metadata df.
        self._metadata(self.path_regis)

        # -- Create bbl dicts.
        bbl_path = os.path.join(self.path_suppl, "12_3_14_bblgrid_clean.npy")
        pluto_mn = os.path.join(self.path_suppl, "pluto", "MN.csv")
        self._bbl_dicts(bbl_path, pluto_mn)

        # -- Load srcs to be ignored.
        null_path = os.path.join(self.path_outpu, "LightCurves", "null_data.pkl")
        self._ign_srcs(null_path)

        # -- Load appertures.
        self._load_wins(os.path.join(self.path_suppl, "window_labels.out"))

        # -- Find apperture centers.
        self._src_centers(bbl_path)

        # -- Load bigoffs if they have been saved to file.
        path_bigoff = os.path.join(self.path_outpu, "LightCurves", "bigoffs.pkl")

        if os.path.isfile(path_bigoff):
            self._load_bigoffs(path_bigoff)
        else:
            self._write_bigoffs(path_bigoff) # Est. Time: 7 mins.lc.


    def _metadata(self, rpath):
        """Load all registration files and concatenate to pandas df. Find
        indices for every day in corresponding on/off/lightcurve files.
        Args:
            rpath (str) - path to registration parent folder.
        """

        tstart = time.time()
        print("LIGHTCURVES: Creating metadata...                              ")
        sys.stdout.flush()

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
            # -- Shift data so each continuous night shares a common date.
            reg.timestamp = reg.timestamp - datetime.timedelta(0, 6 * 3600)
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

        # -- Concatenate metadata dataframe.
        self.meta = pd.concat(dfs).drop_duplicates()

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

        # -- Winter and Summer bigoff times.
        bidx = self.bigoffs.index
        # -- Split by month and only select weekdays.
        self.bigoffs_win = self.bigoffs[(bidx.month >= 9) & (bidx.dayofweek < 5)]
        self.bigoffs_sum = self.bigoffs[(bidx.month < 9) & (bidx.dayofweek < 5)]
        # -- Limit to nights with valid data.
        for season in [self.bigoffs_win, self.bigoffs_sum]:
            season = season[(season.index.isin(self.nights)) &
                            (season.isnull().sum(axis=1) < 3800) &
                            ((~season.isnull()).sum(axis=1) > 1000)]

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
        dimg = self.src_lightc.T
        dimg[dimg == -9999.] = np.nan
        offs = self.src_offs.T

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
        self.bigoffs.columns = self.bigoffs.columns + 1

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def _ign_srcs(self, null_path):
        """"""

        cols = ["dd", "null_sources", "null_percent"]
        # -- Load null path.
        if os.path.isfile(null_path):
            null_df = pd.read_pickle(null_path)
        else:
            tstart = time.time()
            print("LIGHTCURVES: Finding sources that are not in all imgs...   ")
            sys.stdout.flush()

            dd_ign = []
            for dd in self.meta.index.unique():
                self.loadnight(dd)
                dd_ign.append([dd, self.null_sources, self.null_percent])
            null_df = pd.DataFrame(dd_ign, columns=cols).set_index("dd")
            null_df.to_pickle(null_path)

            print("LIGHTCURVES: Complete ({:.2f}s)                           " \
                .format(time.time() - tstart))
            sys.stdout.flush()

        # -- Entirely ignore all days with more than 25% null data.
        null_df25 = null_df[null_df.null_percent < 0.25]
        # -- Extract null sources to be ignored.
        self.src_ignore = list(set(src for l in null_df25.null_sources
            for src in l))
        # -- Nights where null_percent < 0.25.
        self.nights = null_df25.index.values
        self.null_df = null_df
        # # -- Merge null_percent with meta df.
        self.meta = self.meta.merge(null_df[["null_percent"]], left_index=True,
            right_index=True, how="left", validate="many_to_one")


    def _find_err_data(self):
        """Find sources (indices) that are null and total null percentage for a
        given night."""

        # -- Find sources included in frame.
        # -- Calculate the mean over all sources.
        smean = self.src_lightc.mean(axis=0)
        # -- Find sources that are all null (i.e., out of frame).
        outframe = smean == -9999
        if sum(outframe) > 0:
            # -- Find the idx of all such sources.
            null_src = [i for i, x in enumerate(outframe) if x == True]
        else:
            null_src = []

        self.null_sources = null_src
        self.null_percent = (float(sum(sum(self.src_lightc == -9999))) /
                             self.src_lightc.size)


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
        self.mat_srcs = np.zeros((nrow, ncol), dtype=bool)
        self.mat_srcs[buff:-buff, buff:-buff] = np.fromfile(inpath, int) \
            .reshape(nrow - 2 * buff, ncol - 2 * buff) \
            .astype(bool)

        # -- Create label map.
        self.mat_labs = ndm.label(self.mat_srcs)[0]
        # -- Replace sources to be ignored.
        shp1, shp2 = self.mat_labs.shape
        self.mat_labs[np.in1d(self.mat_labs, self.src_ignore) \
            .reshape(shp1, shp2)] = 0
        # -- Use replaced labels to update srcs.
        self.mat_srcs = self.mat_labs > 0
        # -- Re-label
        self.mat_labs = ndm.label(self.mat_srcs)[0]

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
        labrange = range(1, self.mat_labs.max() + 1)
        # -- Find center coordinates.
        xy = ndm.center_of_mass(self.mat_srcs, self.mat_labs, labrange)
        # -- Create dict of labels: (xx, yy)
        self.coords = {self.mat_labs[int(coord[0])][int(coord[1])]:
            (int(coord[1]), int(coord[0])) for coord in xy}

        # -- Find bbl for each source.
        bbls = np.load(bbl_path)
        xx, yy = zip(*self.coords.values())
        coord_bbls = [bbls[yy[ii] - 20][xx[ii] - 20] for ii in range(len(xx))]
        self.coords_bbls = dict(zip(self.coords.keys(), coord_bbls))

        # -- Find higher level bldgclass for all lightsources.
        bbl_arb = {k: self.dd_bldg_n2[v] for k, v in self.dd_bbl_bldg.items()}
        ccs = filter(lambda x: x[1] in bbl_arb.keys(), self.coords_bbls.items())
        self.coords_cls = {k: bbl_arb[v] for k, v in ccs}

        # -- Create dict for bbl to 2nd level class:
        self.dd_bbl_bldg2 = {bbl: self.coords_cls.get(src, -1)
            for src, bbl in self.coords_bbls.items()}

        # -- Create dict for bbl: list of sources.
        self.dd_bbl_srcs = defaultdict(list)
        _ = [self.dd_bbl_srcs[v].append(k) for k, v in self.coords_bbls.items()]

        print("LIGHTCURVES: Complete ({:.2f}s)                               " \
            .format(time.time() - tstart))
        sys.stdout.flush()


    def _bbl_dicts(self, bbl_path, pluto_path):
        """Create dictionaries required for plotting bbl images."""

        # -- Load bbls, and replace 0s.
        bbls = np.load(bbl_path)
        np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)
        # -- Load PLUTO.
        df = pd.read_csv(pluto_path, usecols=["BBL", "BldgClass", "ZipCode"]) \
            .set_index("BBL")
        df = df[[isinstance(ii, str) for ii in df.BldgClass]]

        # -- BBL to zipcode,
        self.dd_bbl_zip = {int(k): v for k, v in df.ZipCode.to_dict().items()}
        # -- Create dict of BBL:BldgClass
        self.dd_bbl_bldg = {bbl: df.loc[bbl].BldgClass for bbl in
            filter(lambda x: x in df.index, np.unique(bbls))}
        # -- Create dict of BldgClass:num.
        uniq_class = np.unique(self.dd_bbl_bldg.values())
        self.dd_bldg_n1 = dict(zip(uniq_class, range(len(uniq_class))))
        self.dd_bldg_n1_r = {v: k for k, v in self.dd_bldg_n1.items()}
        # -- Create dict of BBL:num.
        self.dd_bbl_n = dict(zip(self.dd_bbl_bldg.keys(),
            [self.dd_bldg_n1[v] for v in self.dd_bbl_bldg.values()]))
        # -- Create dict of BldgClass:
        self._create_arbclass_dict()

        # -- BBL to income dictionary.
        # -- Load ACS median income data.
        acs_path = os.path.join(self.path_suppl, "ACS", "ACS_15_5YR_B19013",
            "ACS_15_5YR_B19013_with_ann.csv")
        df = pd.read_csv(acs_path, header=1).iloc[:, 2:4]
        df.columns = ["Geography", "Median_HH_Income"]
        # -- Pull block group and census tract.
        df["BG"] = df.Geography.apply(lambda x: x.split(", ")[0].strip("Block Group "))
        df["CT"] = df.Geography.apply(lambda x: x.split(", ")[1].strip("Census Tract "))
        df.Median_HH_Income.replace("-", 0, inplace=True)
        df.Median_HH_Income.replace("250,000+", 250000, inplace=True)
        df.Median_HH_Income = df.Median_HH_Income.astype(float)
        df.BG = df.BG.astype(int)
        df.CT = df.CT.astype(float)

        # -- Load PLUTO data.
        pluto_path = os.path.join(self.path_suppl, "pluto", "MN.csv")
        pluto = pd.read_csv(pluto_path, usecols=["Block", "CT2010", "CB2010", "BBL"])
        pluto.BBL = pluto.BBL.astype(int)
        # -- Pull block group.
        pluto.CB2010 = pluto.CB2010.astype(str)
        pluto["BG"] = pluto.CB2010.str[0].replace("n", np.nan).astype(float)

        # -- Merge ACS median income and PLUTO data.
        df = pluto.merge(df, left_on=["BG", "CT2010"], right_on=["BG", "CT"], how="left")
        # -- BBL to median income dict:
        self.dd_bbl_income = df[["BBL", "Median_HH_Income"]].set_index("BBL") \
            .to_dict()["Median_HH_Income"]


    def _create_arbclass_dict(self):
        """Categorize BldgClass numbers.
        """

        arbclass = {}
        res = ["B", "C", "D", "N", "R1", "R2", "R3", "R4", "S"]
        com = ["J", "K", "L", "O", "RA", "RB", "RC", "RI"]
        mix = ["RM", "RR", "RX"]
        ind = ["F"]
        mis = ["G", "H", "I", "M", "P", "Q", "T", "U", "V", "W", "Y", "Z"]
        for cc in self.dd_bldg_n1_r.values():
            for v in res:
                if cc.startswith(v):
                    arbclass[cc] = 1
            for v in com:
                if cc.startswith(v):
                    arbclass[cc] = 2
            for v in mix:
                if cc.startswith(v):
                    arbclass[cc] = 3
            for v in ind:
                if cc.startswith(v):
                    arbclass[cc] = 4
            for v in mis:
                if cc.startswith(v):
                    arbclass[cc] = 5

        self.dd_bldg_n2 = arbclass


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
        lc_fname = "light_curves_{}.npy".format(fname)
        ons_fname = "good_ons_{}.npy".format(fname)
        offs_fname = "good_offs_{}.npy".format(fname)
        lc_fpath = os.path.join(self.path_light, lc_fname)
        ons_fpath  = os.path.join(self.path_varia, ons_fname)
        offs_fpath = os.path.join(self.path_varia, offs_fname)

        # -- Load .npy files and select for the correct night.
        dimg = np.load(lc_fpath).mean(-1)[start: end]
        ons = np.load(ons_fpath)[start: end]
        offs = np.load(offs_fpath)[start: end]

        if hasattr(self, "src_ignore"):
            # -- Remove sources that are not in all images.
            dimg = np.delete(dimg, self.src_ignore, axis=1)
            ons = np.delete(ons, self.src_ignore, axis=1)
            offs = np.delete(offs, self.src_ignore, axis=1)

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
            self.src_lightc, self.src_ons, self.src_offs = self._loadfiles(
                fname, start, end)

        if len(metadata) == 2:
            data = []
            for idx in range(2):
                fname = metadata[idx]["fname"]
                start = metadata[idx]["start"]
                end = metadata[idx]["end"]
                data.append(self._loadfiles(fname, start, end))

            data = zip(data[0], data[1])
            self.src_lightc = np.concatenate(data[0], axis=0)
            self.src_ons = np.concatenate(data[1], axis=0)
            self.src_offs = np.concatenate(data[2], axis=0)

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
    lc.loadnight(pd.datetime(2013, 11, 15).date())
