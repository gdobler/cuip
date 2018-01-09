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
    """Gaussian filter (sigma=30) each lightcurve in a given night and min-max.
    Args:
        lc (obj) - LightCurves object.
    Returns:
        minmax (array) - prprocessed array of light curves.
        maks (array) - mask for preprocessed array of light curves.
    """
    # -- Print status.
    tstart = _start("Preprocessing light curves.")
    mask = gf((lc.lcs > -9999).astype(float), (width, 0)) > 0.9999
    msk_sm = gf(mask.astype(float), (width, 0))
    lcs_sm = gf(lc.lcs * mask, (width, 0)) / (msk_sm + (msk_sm == 0))
    minmax = MinMaxScaler().fit_transform(lcs_sm)
    # -- Print status.
    _finish(tstart)
    return [minmax, mask]


def median_detrend(minmax, width=30):
    """Detrend light curves using a gaussian filtered median.
    Args:
        minmax (array) - preprocessed array of light curves.
        width (int; default=30) - gaussian filter width.
    Returns:
        dtrend (array)
    """
    dtrend = minmax.T - gf(np.median(minmax, axis=1), width)
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
    # # -- Pull residential idx coords.
    # res_labs = filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())
    # # -- Subselect bigoffs for residential buildings.
    # res_boffs = np.array(bigoffs)[np.array(res_labs) - 1]
    # # -- Subselect residential minmaxed lightcurves.
    # tmp = minmax.T[np.array(res_labs) - 1]
    # # -- Argsort big residential bigoffs.
    # idx = np.array(res_boffs).argsort()
    tmp = minmax.T
    idx = np.array(bigoffs).argsort()
    # -- Plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(tmp[idx], aspect="auto")
    ax.scatter(np.array(bigoffs)[idx], range(len(bigoffs)), s=3, label="BigOff")
    ax.set_ylim(0, tmp.shape[0])
    ax.set_xlim(0, tmp.shape[1])
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Light Sources")
    ax.set_title("Residential Light Sources w Big Offs ({})".format(lc.night))
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    if show:
        plt.show(block=True)
    else:
        plt.savefig("./pdf/night_{}.png".format(lc.night))
        plt.close("all")
    # -- Print status.
    _finish(tstart)

def main(lc):
    """"""
    minmax, mask = preprocess_lightcurves(lc)
    minmax = median_detrend(minmax)
    lcs_gd = gaussian_differences(minmax, mask)
    avg, sig = sigma_clipping(lcs_gd, iters=10)
    tags_on, tags_off = tag_ons_offs(lcs_gd, avg, sig, sig_peaks=10.)
    good_ons, good_offs = cross_check(minmax, tags_on, tags_off)
    bigoffs = find_bigoffs(minmax, good_offs)
    return [minmax, good_ons, good_offs, bigoffs]

def main_(lc):
    """"""
    minmax, mask = preprocess_lightcurves(lc)
    minmax = median_detrend(minmax)
    lcs_diff = high_pass_subtraction(minmax)
    lcs_gd = gaussian_differences(lcs_diff, mask)
    avg, sig = sigma_clipping(lcs_gd, iters=10)
    tags_on, tags_off = tag_ons_offs(lcs_gd, avg, sig, sig_peaks=10.)
    good_ons, good_offs = cross_check(minmax, tags_on, tags_off)
    bigoffs = find_bigoffs(minmax, good_offs)
    return [minmax, good_ons, good_offs, bigoffs]


if __name__ == "__main__":
    # -- Detect ons/offs and bigoffs and plot the results.
    minmax, good_ons, good_offs, bigoffs = main_(lc)
    plot_bigoffs(minmax, bigoffs)
    # # -- Run all detections.
    # outpath = os.path.join(OUTP, "onsoffs")
    # bigoffs_df = []
    # for dd in lc.meta.index:
    #     lc.loadnight(dd)
    #     minmax, good_ons, good_offs, bigoffs = main(lc)
    #     bigoffs_df.append(bigoffs)
    #     onfname = "good_ons_{}.npy".format(lc.night)
    #     offname = "good_offs_{}.npy".format(lc.night)
    #     np.save(os.path.join(outpath, onfname), good_ons)
    #     np.save(os.path.join(outpath, offname), good_offs)
    #     plot_bigoffs(minmax, bigoffs, False)
    # df = pd.DataFrame(bigoffs_df)
    # df["index"] = lc.meta.index
    # df.to_pickle(os.path.join(outpath, "bigoffs.pkl"))
