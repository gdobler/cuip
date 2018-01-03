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


def preprocess_lightcurves(lc, width=30):
    """Gaussian filter (sigma=30) each lightcurve in a given night. Median
    detrend the resulting lightcurves and min-max.
    Args:
        lc (obj) - LightCurves object.
    Returns:
        minmax (array) - prprocessed array of light curves.
        maks (array) - mask for preprocessed array of light curves.
    """
    tstart = time.time()
    print("LIGHTCURVES: Preprocessing light curves.                           ")
    sys.stdout.flush()
    mask = gf((lc.src_lightc > -9999).astype(float), (width, 0)) > 0.9999
    msk_sm = gf(mask.astype(float), (width, 0))
    lcs_sm = gf(lc.src_lightc * mask, (width, 0)) / (msk_sm + (msk_sm == 0))
    minmax = MinMaxScaler().fit_transform(lcs_sm)
    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))
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


def gaussian_differences(minmax, mask, delta=2):
    """Calculate gaussian differences in light curves.
    Args:
        minmax (array) - preprocessed array of light curves.
        mask (array) - mask for preprocessed array of light curves.
        delta (int; default=2) - delta (offset) value for gaussian diff. calc.
    Returns:
        lcs_gd (array) - array of light curve gaussian differences.
    """
    tstart = time.time()
    print("LIGHTCURVES: Calculating gaussian differences.                     ")
    sys.stdout.flush()
    lcs_gd = np.ma.zeros(minmax.shape, dtype=minmax.dtype)
    lcs_gd[delta // 2: -delta // 2] = minmax[delta:] - minmax[:-delta]
    lcs_gd.mask = np.zeros_like(mask)
    lcs_gd.mask[delta // 2: -delta // 2] = ~(mask[delta:] * mask[:-delta])
    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))
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
    tstart = time.time()
    print("LIGHTCURVES: Sigma clipping.                                       ")
    sys.stdout.flush()
    tmsk = lcs_gd.mask.copy()
    for _ in range(iters):
        avg = lcs_gd.mean(0)
        sig = lcs_gd.std(0)
        lcs_gd.mask = np.abs(lcs_gd - avg) > sig_clip * sig
    lcs_gd.mask = tmsk
    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))
    return [avg, sig]


def tag_ons_offs(lcs_gd, avg, sig, sig_peaks=0.):
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
    tstart = time.time()
    print("LIGHTCURVES: Tag ons and offs.                                     ")
    sys.stdout.flush()
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
    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))
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
    tstart = time.time()
    print("LIGHTCURVES: Cross check ons and offs.                             ")
    sys.stdout.flush()
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
    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))
    return [good_ons, good_offs]


def find_bigoffs(minmax, good_offs):
    """Find big offs from good_offs.
    Args:
        minmax (array) - preprocessed array of light curves.
        good_offs (array) - array of good offs.
    Returns:
        bigoffs (list) - timestep of bigoff for each source.
    """
    tstart = time.time()
    print("LIGHTCURVES: Calculate bigoffs.                                    ")
    sys.stdout.flush()
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
    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))
    return bigoffs


def plot_bigoffs(minmax, bigoffs, show=True):
    """Plot bigoffs."""
    print("Plotting residential bigoffs.")
    sys.stdout.flush()
    # -- Pull residential idx coords.
    res_labs = filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())
    # -- Subselect bigoffs for residential buildings.
    res_boffs = np.array(bigoffs)[np.array(res_labs) - 1]
    # -- Subselect residential minmaxed lightcurves.
    tmp = minmax.T[np.array(res_labs) - 1]
    # -- Argsort big residential bigoffs.
    idx = np.array(res_boffs).argsort()
    # -- Plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(tmp[idx])
    ax.scatter(np.array(res_boffs)[idx], range(len(res_boffs)), s=3, label="BigOff")
    ax.set_ylim(0, tmp.shape[0])
    ax.set_xlim(0, tmp.shape[1])
    ax.set_yticks([])
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Light Sources")
    ax.set_title("Residential Light Sources w Big Offs ({})".format(lc.night))
    ax.legend()
    if show:
        plt.show(block=True)
    else:
        plt.savefig("./pdf/night_{}.png".format(lc.night))

def main(lc):
    """"""
    minmax, mask = preprocess_lightcurves(lc)
    minmax = median_detrend(minmax)
    lcs_gd = gaussian_differences(minmax, mask)
    avg, sig = sigma_clipping(lcs_gd, iters=10)
    tags_on, tags_off = tag_ons_offs(lcs_gd, avg, sig, sig_peaks=3.)
    good_ons, good_offs = cross_check(minmax, tags_on, tags_off)
    bigoffs = find_bigoffs(minmax, good_offs)
    return [minmax, good_ons, good_offs, bigoffs]


if __name__ == "__main__":
    # # -- Detect ons/offs and bigoffs and plot the results.
    # minmax, good_ons, good_offs, bigoffs = main(lc)
    # plot_bigoffs(minmax, bigoffs)
    # -- Run all detections.
    outpath = os.path.join(OUTP, "onsoffs")
    bigoffs_df = []
    for dd in lc.meta.index:
        lc.loadnight(dd)
        minmax, good_ons, good_offs, bigoffs = main(lc)
        bigoffs_df.append(bigoffs)
        onfname = "good_ons_{}.npy".format(lc.night)
        offname = "good_offs_{}.npy".format(lc.night)
        np.save(os.path.join(outpath, onfname), good_ons)
        np.save(os.path.join(outpath, offname), good_offs)
        plot_bigoffs(minmax, bigoffs, False)
    df = pd.DataFrame(bigoffs_df)
    df["index"] = lc.meta.index
    df.to_pickle(os.path.join(outpath, "bigoffs.pkl"))
