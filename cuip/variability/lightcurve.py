from __future__ import print_function

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.ndimage.measurements as ndm


def start(text):
    """Print status of process.
    Args:
        text (str) - Text to print to stdout.
    Returns:
        time.time()
    """
    print("LIGHTCURVES: {}".format(text))
    sys.stdout.flush()
    return time.time()


def finish(tstart):
    """Prints elapsed time from start of process.
    Args:
        tstart (time).
    """
    print("LIGHTCURVES: Complete ({:.2f}s)".format(time.time() - tstart))


class LightCurves(object):
    def __init__(self, path_lig, path_var, path_reg, path_sup, path_out,
        load_all=True):
        """Container for luminosity timeseries.
        Args:
            path_lig (str) - path to light curve directory.
            path_var (str) - path to variability directory (ons/offs).
            path_reg (str) - path to registration directory.
            path_sup (str) - path to supplementary directory.
            path_out (str) - path to output directory.
            load_all (bool, default=True) - load ons/offs and bigoffs.
        """
        # -- Data paths.
        self.path_lig = path_lig
        self.path_var = path_var
        self.path_reg = path_reg
        self.path_sup = path_sup
        self.path_out = path_out
        # -- Create metadata df.
        self._metadata(self.path_reg)
        # -- Load a night.
        self.loadnight(self.meta.index[0], False, False)
        # -- Create data dictionaries.
        wpath = os.path.join(path_sup, "window_labels.out")
        bblpath = os.path.join(path_sup, "12_3_14_bblgrid_clean.npy")
        plutopath = os.path.join(path_sup, "pluto", "MN.csv")
        self._data_dictionaries(wpath, bblpath, plutopath)


    def _metadata(self, path_reg, tspan=[(21, 0), (4, 30)]):
        """Load all registration files and concatenate to pandas df. Find
        indices for every day in corresponding on/off/lightcurve files.
        Args:
            path_reg (str) - path to registration directory.
            tspan (list) - tuples defining span of each day (hour, minute).
        """
        # -- Print status.
        tstart = start("Creating metadata.")
        # -- Create start & end times to subselect.
        stime, etime = [pd.datetime(9, 9, 9, *tt).time() for tt in tspan]
        # -- Store timedelta used to shift times to a single night.
        self.timedelta = pd.to_timedelta(str(etime))
        # -- Empty list to append dfs, and cols to parse.
        dfs  = []
        cols = ["timestamp"]
        # -- Create metadata dfs from each registration file.
        for rfile in sorted(os.listdir(path_reg)):
            # -- Read the current registration file.
            rfilep = os.path.join(path_reg, rfile)
            reg    = pd.read_csv(rfilep, parse_dates=cols, usecols=cols)
            # -- Subselect for start and end times.
            reg = reg[(reg.timestamp.dt.time > stime) |
                      (reg.timestamp.dt.time < etime)].reset_index()
            # -- Shift data so each continuous night shares a common date.
            reg.timestamp = reg.timestamp - self.timedelta
            # -- Create df with start, end, fname, and timesteps of each night.
            gby = reg.groupby(reg.timestamp.dt.date)["index"]
            meta = pd.concat([gby.min(), (gby.max() + 1)], axis=1)
            meta.columns = ["start", "end"]
            meta["fname"] = rfile[-8:-4]
            meta["timesteps"] = meta.end - meta.start
            meta.index = pd.to_datetime(meta.index)
            dfs.append(meta)
        # -- Concatenate metadata dataframe.
        self.meta = pd.concat(dfs)
        # -- Print status
        finish(tstart)


    def _data_dictionaries(self, wpath, bblpath, plutopath):
        """Load the window matrix, label coords, and create relevant data dicts.
        Args:
            wpath (str) - path to window apperture file.
            bblpath (str) - path to bbl map file.
            plutopath (str) path to pluto data file.
        """
        # -- Print status.
        tstart = start("Creating data dictionaries.")
        # -- Create building class data dictionary.
        ind = ["F"]
        mix = ["RM", "RR", "RX"]
        com = ["J", "K", "L", "O", "RA", "RB", "RC", "RI"]
        res = ["B", "C", "D", "N", "R1", "R2", "R3", "R4", "S"]
        mis = ["G", "H", "I", "M", "P", "Q", "T", "U", "V", "W", "Y", "Z"]
        self.dd_bldgclss = {val: ii + 1 for ii, ll in
                            enumerate([res, com, mix, ind, mis]) for val in ll}
        # -- Load light source matrix.
        nrow, ncol, buff = 2160, 4096, 20
        self.matrix_sources = np.zeros((nrow, ncol), dtype=bool)
        self.matrix_sources[buff: -buff, buff:-buff] = np.fromfile(wpath, int) \
            .reshape(nrow - 2 * buff, ncol -2 * buff).astype(bool)
        # -- Create label matrix.
        self.matrix_labels = ndm.label(self.matrix_sources)[0]
        # -- Find coordinates for each light source.
        labels = range(1, self.matrix_labels.max() + 1)
        xy = ndm.center_of_mass(self.matrix_sources, self.matrix_labels, labels)
        self.coords = {self.matrix_labels[int(xx)][int(yy)]: (int(xx), int(yy))
            for xx, yy in xy}
        # -- Find bbl corresponding to each window apperture.
        bbls = np.load(bblpath)
        np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)
        xx, yy = zip(*self.coords.values())
        coord_bbls = [bbls[xx[ii] - 20][yy[ii] - 20] for ii in range(len(xx))]
        self.coords_bbls = dict(zip(self.coords.keys(), coord_bbls))
        # -- Map BBL to ZipCode.
        df = pd.read_csv(plutopath, usecols=["BBL", "BldgClass", "ZipCode"]) \
            .set_index("BBL")
        df = df[[isinstance(ii, str) for ii in df.BldgClass]]
        self.dd_bbl_zip = {int(k): v for k, v in df.ZipCode.to_dict().items()}
        # -- Map BBL to building class.
        self.dd_bbl_bldgclss = {int(k): v for k, v in df.BldgClass.to_dict().items()}
        # -- Map coordinates to building class.
        crd_cls = {k: self.dd_bbl_bldgclss.get(v, -9999)
                    for k, v in self.coords_bbls.items()}
        self.coords_cls = {k: self.dd_bldgclss[[kk for kk in
                                                self.dd_bldgclss.keys()
                                                if v.startswith(kk)][0]]
                           for k, v in crd_cls.items()
                           if v != -9999}
        # -- Print status.
        finish(tstart)


    def _loadfiles(self, fname, idx_start, idx_end, load_all=True, lc_mean=True,
        lc_dtrend=False):
        """Helper function to load lcs, ons, and offs from file.
        Args:
            fname (str) - fname suffix (e.g., '0001').
            idx_start (int) - idx for the start of the evening.
            idx_end (int) - idx for the end of the evening.
            load_all (bool, default=True) - load ons and offs?
            lc_mean (bool, default=True) - take the mean across color channels?
            lc_dtrend (bool, default=False) - load detrended lightcurves?
        Returns:
            (list) - [lcs, ons, off]
        """
        # -- Load lightcurves.
        if lc_dtrend:
            path = os.path.join(self.path_var, "detrended_{}.npy".format(self.night.date()))
            lcs = np.load(path)
        else:
            path = os.path.join(self.path_lig, "light_curves_{}.npy".format(fname))
            lcs = np.load(path)[idx_start: idx_end]
            if lc_mean:
                lcs = lcs.mean(-1)
        # -- Load transitions (unless detecting ons/offs).
        if load_all:
            fname = "good_ons_{}.npy".format(self.night.date())
            ons = np.load(os.path.join(self.path_var, fname))
            fname = "good_offs_{}.npy".format(self.night.date())
            off = np.load(os.path.join(self.path_var, fname))
        else: # -- Else set to empty lists.
            ons, off = [], []
        # -- Return loaded data.
        return [lcs, ons, off]


    def loadnight(self, night, load_all=True, lc_mean=True, lc_dtrend=False):
        """Uses loadfile to load a set of lightcurves, ons, and offs, and
        formats data if data is stored over multiple files.
        Args:
            night (datetime) - night to load data for.
            load_all (bool, default=True) - load ons and offs?
            lc_mean (bool, default=True) - take the mean across color channels?
            lc_dtrend (bool, default=False) - load detrended lightcurves?
        """
        # -- Print status.
        tstart = start("Loading data for {}.".format(night.date()))
        # -- Store night.
        self.night = night
        # -- Get records for given night from metadata.
        mdata = self.meta[self.meta.index == night].to_dict("records")
        # -- If there are no records for the provided night, raise an error.
        if len(mdata) == 0:
            raise ValueError("{} is not a valid night.".format(night.date()))
        else:
            data = []
            # -- For each record in the metadata table... (There may be multiple
            # -- records if the data is spread over multiple files).
            for idx in range(len(mdata)):
                # -- Grab start idx, end idx, fname, and timesteps.
                vals = mdata[idx]
                idx_start, idx_end, fname = vals["start"], vals["end"], vals["fname"]
                # -- Load lcs, ons, and offs and append to data.
                data.append(self._loadfiles(fname, idx_start, idx_end, load_all, lc_mean, lc_dtrend))
            # -- Concatenate all like files and store.
            self.lcs = np.concatenate([dd[0] for dd in data], axis=0)
            self.lc_ons = np.concatenate([dd[1] for dd in data], axis=0)
            self.lc_offs = np.concatenate([dd[2] for dd in data], axis=0)
            if lc_mean: # -- Only backfill/forward fill if taking mean.
                # -- Backfill/forward fill -9999 in lcs (up to 3 consecutive).
                self.lcs = pd.DataFrame(self.lcs).replace(-9999, np.nan) \
                    .fillna(method="bfill", limit=3, axis=0) \
                    .fillna(method="ffill", limit=3, axis=0) \
                    .replace(np.nan, -9999).as_matrix()
        # -- Load bigoff dataframe if load_all has been specified.
        if load_all:
            path_boffs = os.path.join(self.path_var, "bigoffs.pkl")
            self.lc_bigoffs = pd.read_pickle(path_boffs).set_index("index")
        # -- Print status
        finish(tstart)


if __name__ == "__main__":
    # -- Load environmental variables.
    LIGH = os.environ["LIGHTCURVES"]
    VARI = os.environ["VARIABILITY"]
    REGI = os.environ["REGISTRATION"]
    SUPP = os.environ["SUPPPATH"]
    OUTP = os.environ["OUTPATH"]
    # -- Revised paths to load histogram matched lightcurves and corresponding ons/offs.
    LIGH = os.path.join(OUTP, "histogram_matching")
    VARI = os.path.join(OUTP, "onsoffs")
    # -- Create LightCurve object.
    lc = LightCurves(LIGH, VARI, REGI, SUPP, OUTP)
