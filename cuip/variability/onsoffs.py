from __future__ import print_function

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import correlate1d
from scipy.ndimage.filters import gaussian_filter as gf

plt.style.use("ggplot")

def _start(text):
    """"""
    print("LIGHTCURVES: {}".format(text))
    sys.stdout.flush()
    return time.time()


def _finish(tstart):
    """"""
    print("LIGHTCURVES: Complete ({:.2f}s)".format(time.time() - tstart))


def preprocess_lightcurves(lc, width=30):
    """Gaussian filter (sigma=30) each lightcurve in a given night.
    Args:
        lc (obj) - LightCurves object.
    Returns:
        lcs_sm (array) - prprocessed array of light curves.
        maks (array) - mask for preprocessed array of light curves.
    """
    # -- Print status.
    tstart = _start("Preprocessing light curves.")
    mask = gf((lc.lcs > -9999).astype(float), (width, 0)) > 0.9999
    msk_sm = gf(mask.astype(float), (width, 0))
    lcs_sm = gf(lc.lcs * mask, (width, 0)) / (msk_sm + (msk_sm == 0))
    # -- Print status.
    _finish(tstart)
    return [lcs_sm, mask]


def min_max_lightcurves(lcs):
    """Minmax lightcurves.
    Args:
        lcs (array) - light curves.
    Returns:
        minmax (array) - preprocessed array of light curves.
    """
    # -- Print status.
    tstart = _start("Minmaxing light curves.")
    minmax = MinMaxScaler().fit_transform(lcs)
    # -- Print status.
    _finish(tstart)
    return(minmax)


def median_detrend(minmax, width=30):
    """Detrend light curves using a gaussian filtered median.
    Args:
        minmax (array) - preprocessed array of light curves.
        width (int; default=30) - gaussian filter width.
    Returns:
        dtrend (array)
    """
    # -- Print status.
    tstart = _start("Detrending light curves.")
    med = np.median(minmax, axis=1)
    mask = gf((med > 0).astype(float), width) > 0.9999
    msk_sm = gf(mask.astype(float), width)
    med_sm = gf(med * mask, width) / (msk_sm + (msk_sm == 0))
    dtrend = minmax.T - med_sm
    # -- Print status.
    _finish(tstart)
    return dtrend.T


def high_pass_subtraction(minmax, width=360):
    """"""
    # -- Print status.
    tstart = _start("High pass subtraction.")
    lcs_hp = gf(minmax, (width, 0))
    lcs_diff = minmax - lcs_hp
    # -- Print status.
    _finish(tstart)
    return lcs_diff


def gaussian_differences(minmax, mask, delta=2):
    """Calculate gaussian differences in light curves.
    Args:
        minmax (array) - preprocessed array of light curves.
        mask (array) - mask for preprocessed array of light curves.
        delta (int; default=2) - delta (offset) value for gaussian diff. calc.
    Returns:
        lcs_gd (array) - array of light curve gaussian differences.
    """
    # -- Print status.
    tstart = _start("Calculating gaussian differences.")
    lcs_gd = np.ma.zeros(minmax.shape, dtype=minmax.dtype)
    lcs_gd[delta // 2: -delta // 2] = minmax[delta:] - minmax[:-delta]
    lcs_gd.mask = np.zeros_like(mask)
    lcs_gd.mask[delta // 2: -delta // 2] = ~(mask[delta:] * mask[:-delta])
    # -- Print status.
    _finish(tstart)
    return lcs_gd


def sigma_clipping(lcs_gd, sig_clip=2, iters=10):
    """Sigma clip gaussian differences.
    Args:
        lcs_gd (array) - array of light curve gaussian differences.
        sig_clip (int; default=2) - sigma value for clipping.
        iters (int; default=10) - number of sigma clipping iterations.
    Returns:
        avg (array) - average values.
        sig (array) - standard deviation values.
    """
    # -- Print status.
    tstart = _start("Sigma clipping.")
    tmsk = lcs_gd.mask.copy()
    for _ in range(iters):
        avg = lcs_gd.mean(0)
        sig = lcs_gd.std(0)
        lcs_gd.mask = np.abs(lcs_gd - avg) > sig_clip * sig
    lcs_gd.mask = tmsk
    # -- Print status.
    _finish(tstart)
    return [avg, sig]


def tag_ons_offs(lcs_gd, avg, sig, sig_peaks=10.):
    """Tag potential ons and offs.
    Args:
        lcs_gd (array) - array of light curve gaussian differences.
        avg (array) - average values.
        sig (array) - standard deviation values.
        sig_peaks (int; defaul=0.)
    Returns:
        tags_on (array) - array of potential ons.
        tags_offs (array) - array of potential offs.
    """
    # -- Print status.
    tstart = _start("Tag ons and offs.")
    tags_on = np.zeros(lcs_gd.shape, dtype=bool)
    tags_off = np.zeros(lcs_gd.shape, dtype=bool)
    # -- Pos values, higher than prev and next value, and are not masked.
    tags_on[1:-1] = (lcs_gd - avg > sig_peaks * sig)[1:-1] & \
                     (lcs_gd[1:-1] > lcs_gd[2:]) & \
                     (lcs_gd[1:-1] > lcs_gd[:-2]) & \
                     ~lcs_gd.mask[1:-1]
    # -- Neg values, lower than prev and next values, and are not masked.
    tags_off[1:-1] = (lcs_gd - avg < -sig_peaks * sig)[1:-1] & \
                     (lcs_gd[1:-1] < lcs_gd[2:]) & \
                     (lcs_gd[1:-1] < lcs_gd[:-2]) & \
                     ~lcs_gd.mask[1:-1]
    # -- Print status.
    _finish(tstart)
    return [tags_on, tags_off]


def cross_check(minmax, tags_on, tags_off, width=30, sig=2):
    """Cross check potential ons and offs for noise.
    Args:
        minmax (array) - preprocessed array of light curves.
        tags_on (array) - array of potential ons.
        tags_offs (array) - array of potential offs.
        width (int; default=30) - width to check for noise.
        sig (int; defualt=2) - significance level for detections.
    Returns:
        good_ons (array) - array of good ons.
        good_offs (array) - array of good offs.
    """
    # -- Print status.
    tstart = _start("Cross check ons and offs.")
    # -- Set up filters
    mean_diff = ((np.arange(2 * width) >= width) * 2 - 1) / float(width)
    mean_left = 1.0 * (np.arange(2 * width) < width) / float(width)
    mean_right = 1.0 * (np.arange(2 * width) >= width) / float(width)
    # -- Calculate mean difference across transitions
    lcs_md = np.abs(correlate1d(minmax, mean_diff, axis=0))
    # -- Calculate max standard deviation across transitions.
    lcs_sq = minmax ** 2
    lcs_std = np.sqrt(np.maximum(correlate1d(lcs_sq, mean_left, axis=0) -
                                 correlate1d(minmax, mean_left, axis=0) ** 2,
                                 correlate1d(lcs_sq, mean_right, axis=0) -
                                 correlate1d(minmax, mean_right, axis=0) ** 2))
    good_arr = lcs_md > sig * lcs_std
    good_ons = tags_on & good_arr
    good_offs = tags_off & good_arr
    # -- Print status.
    _finish(tstart)
    return [good_ons, good_offs]


def find_bigoffs(minmax, good_offs):
    """Find big offs from good_offs.
    Args:
        minmax (array) - preprocessed array of light curves.
        good_offs (array) - array of good offs.
    Returns:
        bigoffs (list) - timestep of bigoff for each source.
    """
    # -- Print status.
    tstart = _start("Calculate bigoffs.")
    bigoffs = []
    # -- For each lightcurve and corresponding set of offs.
    for src, offs in zip(minmax.T, good_offs.T):
        # -- Zero placeholder if there isn't a detected off.
        if sum(offs) < 1:
            bigoffs.append(np.nan)
        else:
            bigoff = (0, -9999)
            # -- Pull idx for all offs.
            idx = [ix for ix, boo in enumerate(offs) if boo == True]
            # -- For each idx, check mean before and after.
            for ii in idx:
                mm = np.nanmean(src[:ii]) - np.nanmean(src[ii:])
                # -- Keep max.
                if mm > bigoff[1]:
                    bigoff = (ii, mm)
            bigoffs.append(bigoff[0])
    # -- Print status.
    _finish(tstart)
    return bigoffs


def plot_bigoffs(minmax, bigoffs, show=True):
    """Plot bigoffs."""
    # -- Print status.
    tstart = _start("Plotting residential bigoffs.")
    # -- Argsort by bigoff time, to sort plot.
    idx = np.array(bigoffs).argsort()
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    # -- Imshow sorted by bigoffs.
    ax.imshow(minmax.T[idx], aspect="auto")
    # -- Scatter bigofff times.
    ax.scatter(np.array(bigoffs)[idx], range(len(bigoffs)), s=3, label="BigOff")
    # -- Formatting.
    ax.set_ylim(0, minmax.T.shape[0])
    ax.set_xlim(0, minmax.T.shape[1])
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["21:00", "22:00", "23:00", "24:00" ,"1:00", "2:00", "3:00", "4:00"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Light Sources")
    ax.set_title("Light Sources w Big Offs ({})" \
        .format(lc.night.date()))
    ax.legend()
    ax.grid(False)
    ax.xaxis.grid(True, color="w", alpha=0.2)
    plt.tight_layout()
    # -- Show or save.
    if show:
        plt.show(block=True)
    else:
        if not os.path.exists("./pdf/"):
            os.mkdir("./pdf")
        plt.savefig("./pdf/night_{}.png".format(lc.night.date()))
        plt.close("all")
    # -- Print status.
    _finish(tstart)


def main(lc):
    """"""
    lcs_sm, mask = preprocess_lightcurves(lc)
    minmax = min_max_lightcurves(lcs_sm)
    minmax = median_detrend(minmax)
    lcs_diff = high_pass_subtraction(minmax)
    lcs_gd = gaussian_differences(lcs_diff, mask)
    avg, sig = sigma_clipping(lcs_gd, iters=10)
    tags_on, tags_off = tag_ons_offs(lcs_gd, avg, sig, sig_peaks=10.)
    good_ons, good_offs = cross_check(minmax, tags_on, tags_off)
    bigoffs = find_bigoffs(minmax, good_offs)
    return [minmax, good_ons, good_offs, bigoffs]


class CLI(object):
    def __init__(self):
        """"""
        try: # -- Check if light curve object exists (named)
            lc
        except: # -- Raise error if not.
            raise NameError("LIGHTCURVES: lc (obj) is not defined.")
        else: # -- Run CLI if it exists.
            # -- Get user input.
            text = ("LIGHTCURVES: Select from options below:\n" +
                    "    [0] Show plot for current night.\n" +
                    "    [1] Write ons/offs to file for all nights. \n"
                    "Selection: ")
            resp = raw_input(text)
            # -- Run plotting for current night if chosen.
            if int(resp) == 0:
                self.one_off()
            # -- Write ons/offs to file for all nights if chosen.
            elif int(resp) == 1:
                self.write_files()
            else: # -- Else alert user of invalid entry, and recurse.
                print("LIGHTCURVES: '{}' is an invalid entry.".format(resp))
                CLI()


    def one_off(self):
        """Calculate values for night current loaded in lc."""
        minmax, good_ons, good_offs, bigoffs = main(lc)
        plot_bigoffs(minmax, bigoffs)


    def write_files(self):
        """Write ons/offs/bigoffs to file for all nights."""
        # -- Define output path.
        outpath = os.path.join(OUTP, "onsoffs")
        print("LIGHTCURVES: Files to be written to {}".format(outpath))
        sys.stdout.flush()
        # -- Empty list to save bigoff dfs.
        bigoffs_df = []
        minmaxs = []
        # -- Loop over each unique day in lc.meta.index.
        for dd in lc.meta.index.unique():
            lc.loadnight(dd, load_all=False)
            minmax, good_ons, good_offs, bigoffs = main(lc)
            minmaxs.append(minmax)
            bigoffs_df.append(bigoffs)
            onfname = "good_ons_{}.npy".format(lc.night.date())
            offname = "good_offs_{}.npy".format(lc.night.date())
            np.save(os.path.join(outpath, onfname), good_ons)
            np.save(os.path.join(outpath, offname), good_offs)
            plot_bigoffs(minmax, bigoffs, False)
        df = pd.DataFrame(bigoffs_df)
        df["index"] = lc.meta.index.unique()
        df.to_pickle(os.path.join(outpath, "bigoffs.pkl"))
        np.save(os.path.join(outpath, "minmax.npy"), np.array(minmaxs))


if __name__ == "__main__":
    CLI()
