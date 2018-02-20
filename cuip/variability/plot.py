from __future__ import print_function

import os
import re
import scipy
import imageio
import cPickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import scipy.ndimage.measurements as ndm
from sklearn.cluster import KMeans
from scipy.misc import factorial
from scipy.optimize import curve_fit
from scipy.stats.stats import linregress
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter as gf
from mpl_toolkits.axes_grid1 import make_axes_locatable
# -- CUIP Imports
from lightcurve import start, finish

plt.style.use("ggplot")


def read_img(fpath):
    """Read .raw or .png image file.
    Args:
        fpath (str) - path to raw image.
    Returns:
        img (array) - image as np array.
    """
    # -- If raw image load and reshape.
    if fpath.endswith(".raw"):
        img = np.fromfile(fpath, dtype=np.uint8).reshape(2160, 4096, 3)[...,::-1]
    # -- If png image just load.
    if fpath.endswith(".png"):
        img = imageio.imread(fpath)
    return img


def adjust_img(lc, img_path):
    """Load img and adjust (roll horizontally and vertically).
    Args:
        lc (obj) - LightCurve object.
        img_path (str) - path to example image.
    Returns:
        img (np) - 2d-array of adjusted image.
    """
    # -- Find drow and dcol for the example image.
    spath = img_path.split("/")
    yyyy, mm, dd = int(spath[5]), int(spath[6]), int(spath[7])
    fnum = lc.meta.loc[pd.datetime(yyyy, mm, dd - 1)]["fname"]
    reg = pd.read_csv(os.path.join(lc.path_reg, "register_{}.csv".format(fnum)))
    img_reg = reg[reg.fname.str[:-4] == os.path.basename(img_path)[:-4]]
    drow = img_reg.drow.values.round().astype(int)[0]
    dcol = img_reg.dcol.values.round().astype(int)[0]
    # -- Print status.
    _ = start("drow {}".format(drow))
    _ = start("dcol {}".format(dcol))
    # -- Roll image.
    img = np.roll(read_img(img_path)[20:-20, 20:-20], (drow, dcol), (0, 1))
    return img


def bbl_income(lc):
    """"""
    # -- Load income data.
    fpath = os.path.join(lc.path_sup, "ACS", "ACS_15_5YR_B19013", "ACS_15_5YR_B19013_with_ann.csv")
    income = pd.read_csv(fpath, header=1).iloc[:, 2:4]
    income.columns = ["Geography", "Median_HH_Income"]
    # -- Pull block group and census tract.
    income["BG"] = income.Geography.apply(lambda x: x.split(", ")[0].strip("Block Group "))
    income["CT"] = income.Geography.apply(lambda x: x.split(", ")[1].strip("Census Tract "))
    income.Median_HH_Income.replace("-", 0, inplace=True)
    income.Median_HH_Income.replace("250,000+", 250000, inplace=True)
    income.Median_HH_Income = income.Median_HH_Income.astype(float)
    income.BG = income.BG.astype(int)
    income.CT = income.CT.astype(float)
    # -- Load PLUTO data.
    fpath = os.path.join(lc.path_sup, "pluto", "MN.csv")
    pluto = pd.read_csv(fpath, usecols=["Block", "CT2010", "CB2010", "BBL"])
    pluto.BBL = pluto.BBL.astype(int)
    pluto.CB2010 = pluto.CB2010.astype(str)
    pluto["BG"] = pluto.CB2010.str[0].replace("n", np.nan).astype(float)
    df = pluto.merge(income, left_on=["BG", "CT2010"], right_on=["BG", "CT"], how="left")
    return df


def plot_bigoffs(lc, lcs, bigoffs, show=True, fname="./pdf/night_{}.png"):
    """Single panel plot, to show lightcurves in various forms sorted by and
    overlain with their bigoff time.
    Args:
        lc (obj) - LightCurve object (accesses lc.night).
        lcs (arr) - Lightcurves array where .shape is (obs, sources).
        bigoffs (arr) - Bigoffs for the given night.
        show (bool, default=True) - Show image or suppress and save.
        fname (str) - fname if saving.

    Example use: plot_bigoffs(lc, lc.lcs, lc.bigoffs.loc[lc.night])
    """
    # -- Print status.
    tstart = start("Plotting bigoffs.")
    # -- Argsort by bigoff time, to sort plot.
    idx = np.array(bigoffs).argsort()
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    # -- Imshow lcs sorted by bigoffs and scatter bigoffs.
    ax.imshow(lcs.T[idx], aspect="auto")
    ax.scatter(np.array(bigoffs)[idx], range(len(bigoffs)), s=3, label="Big Off")
    # -- Plot formatting.
    ax.set_title("Light Sources w Big Offs ({})".format(lc.night.date()))
    ax.set_ylabel("Light Sources")
    ax.set_xlabel("Time")
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
    ax.set_ylim(0, lcs.T.shape[0])
    ax.set_xlim(0, lcs.T.shape[1])
    ax.grid(False)
    ax.xaxis.grid(True, color="w", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    # -- Show or save.
    if show:
        plt.show(block=True)
    else:
        if not os.path.exists("./pdf/"):
            os.mkdir("./pdf")
        plt.savefig(fname.format(lc.night.date()))
        plt.close("all")


def plot_lightcurve_line(lc, idx):
    """Plot single lightcurve from the loaded evening and overlay with all ons,
    offs, and the bigoff.
    Args:
        lc (obj) - LightCurve object.
        idx (int) - source index to plot.
    """
    # -- Print status.
    tstart = start("Plotting individual lightcurve.")
    # -- Collect plotting data.
    lightc = lc.lcs[:, idx]
    ons    = np.argwhere(lc.lc_ons[:, idx] == True).flatten()
    offs   = np.argwhere(lc.lc_offs[:, idx] == True).flatten()
    bigoff = lc.bigoffs.loc[lc.night].loc[idx]
    # -- Create plot and plot values.
    fig, ax = plt.subplots(figsize=(9, 3))
    li  = ax.plot(lightc, c="k", alpha=0.6, label="Lightcurve")
    off = [ax.axvline(off, c="orange", alpha=0.8, label="Off") for off in offs]
    on  = [ax.axvline(on, c="g", alpha=0.8, label="On") for on in ons]
    big = [ax.axvline(boff, c="r", alpha=0.8, label="Big Off")
           for boff in [bigoff] if not np.isnan(boff)]
    # -- Format plots.
    ax.set_title("Lightcurve for Source {} on {}".format(idx, lc.night.date()))
    ax.set_ylabel("Intensity [arb units]")
    ax.set_xlabel("Time")
    ax.set_xlim(0, len(lightc))
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
    ax.set_yticks([])
    plt.legend(handles=[ii[0] for ii in [li, off, on, big] if len(ii) > 0])
    plt.tight_layout()
    plt.show(block=True)


def plot_winter_summer_bigoffs_boxplot(lc):
    """Plot boxplots comparing bigoffs for summer and winter observations for
    residential sources. Current bigoffs were calculated with a wider range of
    dates than we've used elsewhere. Accordingly, the winter tail appears
    longer. This would be a result of astronomical dawn effecting summer vals.
    Args:
        lc (obj) - LightCurve object.
    """
    # -- Print status.
    tstart = start("Plotting winter/summer bigoffs boxplot.")
    # -- List residential source indices.
    res = np.array(filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())) - 1
    # -- Split up winter/summer residential bigoff data, filter nans.
    winter = lc.bigoffs[lc.bigoffs.index.month > 8][res]
    winter = winter.values.ravel()[~np.isnan(winter.values.ravel())]
    summer = lc.bigoffs[lc.bigoffs.index.month < 8][res]
    summer = summer.values.ravel()[~np.isnan(summer.values.ravel())]
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.boxplot([winter, summer], vert=False, positions=[0, 0.2], labels=["Winter", "Summer"])
    # -- Format plot.
    ax.set_title("Median Weekday Res. Bigoff Timestep for Summer and Winter")
    ax.set_xlabel("Time")
    ax.set_ylim(-0.1, 0.3)
    ax.set_xticks(np.array(range(9)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 30)])
    plt.tight_layout()
    plt.show(block=True)


def plot_winter_summer_bigoffs_histrogram(lc, dist_name="burr"):
    """Plot histograms of summer and winter bigoffs for residential sources.
    Args:
        lc (obj) - LightCurve object.
        dist_name (str, defaul="burr") - scipy.stats distribution name.
    """
    # -- Print status.
    tstart = start("Plotting winter/summer bigoffs histograms.")
    # -- List residential source indices.
    res = np.array(filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())) - 1
    # -- Split up winter/summer residential bigoff data, filter nans.
    winter = lc.bigoffs[lc.bigoffs.index.month > 8][res]
    winter = winter.values.ravel()[~np.isnan(winter.values.ravel())]
    summer = lc.bigoffs[lc.bigoffs.index.month < 8][res]
    summer = summer.values.ravel()[~np.isnan(summer.values.ravel())]
    # -- Histogram vars.
    x_bins = np.arange(0, 3000, 100)
    x_mids = x_bins[1:] - 50
    # -- Create plot.
    results = []
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, boffs, cc in [[ax1, winter, "#56B4E9"], [ax2, summer, "#E69F00"]]:
        ax.hist(boffs, x_bins, normed=True, color="gray")
        ax.axvline(boffs.mean(), c="k", ls="dashed")
        ax.axvline(np.median(boffs), c="k", ls=":")
        dist  = getattr(scipy.stats, dist_name)
        param = dist.fit(boffs)
        pdf   = dist.pdf(x_mids, *param)
        results.append(pdf)
        for xx in [ax, ax3]:
            xx.axvline(dist.mean(*param), c=cc, ls="dashed")
            xx.axvline(dist.median(*param), c=cc, ls=":")
            xx.plot(x_mids, pdf, c=cc)
    # -- Compare values of summer v. winter bigoffs.
    ks = scipy.stats.ks_2samp(winter, summer)
    mw = scipy.stats.mannwhitneyu(winter, summer)
    en = scipy.stats.entropy(np.random.choice(winter, len(summer), False), summer)
    # -- Compare fitted models of summer v. winter bigoffs.
    ksm = scipy.stats.ks_2samp(*results)
    mwm = scipy.stats.mannwhitneyu(*results)
    enm = scipy.stats.entropy(*results)
    # -- Text
    for ax, vv in zip([ax1, ax3], [[ks, mw, en], [ksm, mwm, enm]]):
        ax.text(20, 0.00007, "KS-test (p-val): {:.4f}".format(vv[0].pvalue), size=8)
        ax.text(20, 0.00004, "Mann-Whitney (p-val): {:.4f}".format(vv[1].pvalue), size=8)
        ax.text(20, 0.00001, "Entropy: {:.4f}".format(vv[2]), size=8)
    ax1.text(20, 0.0001, "Histogram comparison:", size=8)
    ax3.text(20, 0.0001, "Model comparison:", size=8)
    # -- Format plots.
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.set_xlim(0, 3000)
        ax.set_ylim(0, 0.00085)
        ax.set_xticks(np.array(range(5)) * 720)
        labs = ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 30, 2)])
        labs[0].set_horizontalalignment("left")
        labs[-1].set_horizontalalignment("right")
    ax1.set_ylabel("Counts (Normed)")
    ax1.set_title("Winter Bigoffs")
    ax2.set_title("Summer Bigoffs")
    ax3.set_title("Fitted {} Distributions".format(dist_name.title()))
    ax3.legend([plt.axvline(-1, ls=ii, c="gray") for ii in ["-", "--", ":"]],
        ["Mean", "Median", "Fitted {} Dist.".format(dist_name.title())])
    plt.tight_layout()
    plt.show(block=True)


def plot_source_locations(lc, idx, impath=os.environ["EXIMG"], circles=True,
    windows=True):
    """Plot source locations and appertures on an example image.
    Args:
        lc (obj) - LightCurves object.
        idx (list) - list of source indexes.
        impath (str) - path to image.
        circles (bool) - plot circles to highlight plotted windows?
        windows (bool) - plot appertures?
    """
    # -- Print status.
    tstart = start("Plotting {} source(s) on example image.".format(len(idx)))
    # -- Pull coordinates for sources.
    yy, xx = zip(*[lc.coords[nn] for nn in idx if nn in lc.coords.keys()])
    # -- Load adjusted image.
    img = adjust_img(lc, impath)
    win = np.ma.array(lc.matrix_labels, mask=~np.isin(lc.matrix_labels, idx))
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)
    if windows:
        ax.imshow(win[20:, 20:])
    if circles:
        ax.scatter(np.array(xx) - 20, np.array(yy) - 20, facecolors="none",
                   edgecolors="#E69F00", linewidth=2)
    # -- Format plot.
    ax.set_xlim(0, img.shape[1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show(block=True)


def plot_bbls(lc, alpha=1, scatter=False, background=False,
    impath=os.environ["EXIMG"], cmap="viridis"):
    """Plot bbls with option to show background image.
    Args:
        lc (obj) - LightCurve object.
        alpha (float) - alpha for plotting bbls.
        scatter (bool) - plot sources?
        background (bool) - plot example image as background?
        impath (str) - path to example image.
        cmap (str) - matplotlib colormap to use.
    """
    # -- Print status.
    tstart = start("Plotting bbls.")
    # -- Load bbls, and replace 0s.
    bbls = np.load(os.path.join(lc.path_sup, "12_3_14_bblgrid_clean.npy"))
    bbls = np.ma.array(bbls, mask=bbls == 0)
    # -- Convert coords from dict to x and y lists.
    yy, xx = zip(*lc.coords.values())
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(16, 8))
    if background:
        ax.imshow(adjust_img(lc, impath))
    ax.imshow(bbls, alpha=alpha, cmap=cmap, vmax=1013380040)
    if scatter:
        ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", s=3, cmap=cmap)
    # -- Plot formatting.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    ax.set_facecolor("w")
    plt.tight_layout()
    plt.show(block=True)


def plot_bldgclass(lc, alpha=1, scatter=False, background=False,
    impath=os.environ["EXIMG"], cmap="viridis"):
    """Plot PLTUO BldgClass with option to show background image.
    Args:
        lc (obj) - LightCurve object.
        alpha (float) - alpha for plotting bbls.
        scatter (bool) - plot sources?
        background (bool) - plot example image as background?
        impath (str) - path to example image.
        cmap (str) - matplotlib colormap to use.
    """
    # -- Print status.
    tstart = start("Plotting PLUTO BldgClss (est. time 2 minutes).")
    # -- Load bbls and mask 0s.
    bbls = np.load(os.path.join(lc.path_sup, "12_3_14_bblgrid_clean.npy"))
    bbls = np.ma.array(bbls, mask=bbls == 0)
    # -- Convert coords from dict to x and y lists.
    yy, xx = zip(*lc.coords.values())
    # -- Map bbl to building class.
    clss = sorted(np.unique(lc.dd_bbl_bldgclss.values()))
    bldgclss = [lc.dd_bbl_bldgclss.get(ii, np.nan) for ii in bbls.data.ravel()]
    bldgclss = [clss.index(ii) if ii in clss else np.nan for ii in bldgclss]
    bldgclss = np.array(bldgclss).reshape(*bbls.shape)
    vals = [val for val in np.unique(bldgclss) if not np.isnan(val)]
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(16, 8))
    if background:
        ax.imshow(adjust_img(lc, impath))
    im = ax.imshow(bldgclss, interpolation="none", cmap=cmap, alpha=alpha)
    if scatter:
        ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", s=3, cmap=cmap)
    # -- Create legend.
    colors = [im.cmap(im.norm(val)) for val in np.unique(bldgclss) if not np.isnan(val)]
    patches = [mpatches.Patch(color=colors[ii], label=clss[int(val)])
               for ii, val in enumerate(vals)]
    ax.legend(handles=patches, ncol=16, fontsize=7)
    # -- Format plot.
    ax.set_facecolor("w")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    plt.tight_layout()
    plt.show(block=True)


def plot_higher_level_bldgclss(lc, alpha=1, scatter=False, background=False,
    impath=os.environ["EXIMG"], cmap="viridis"):
    """Plot higher level classification with option to show background image.
    Args:
        lc (obj) - LightCurve object.
        alpha (float) - alpha for plotting bbls.
        scatter (bool) - plot sources?
        background (bool) - plot example image as background?
        impath (str) - path to example image.
        cmap (str) - matplotlib colormap to use.
    """
    # -- Print status.
    tstart = start("Plotting higher level BldgClss.")
    # -- Load bbls, and mask nans.
    # -- ...bldgclss.npy is made using bbls_to_bldgclss.
    bbls = np.load(os.path.join(lc.path_sup, "12_3_14_bblgrid_clean_bldgclss.npy"))
    bbls = np.ma.array(bbls, mask=np.isnan(bbls))
    # -- Convert coords from dict to x and y lists.
    yy, xx = zip(*lc.coords.values())
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(16, 8))
    if background:
        ax.imshow(adjust_img(lc, impath))
    im = ax.imshow(bbls, interpolation="none", cmap=cmap, alpha=alpha)
    if scatter:
        ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", s=3, cmap=cmap)
    # -- Create legend.
    labs = ["Residential", "Commmercial", "Mixed Use", "Industrial", "Misc."]
    colors = [im.cmap(im.norm(val)) for val in range(1, 6)]
    patches = [mpatches.Patch(color=colors[ii], label="{}".format(labs[ii])) for ii in range(5)]
    plt.legend(handles=patches, ncol=17)
    # -- Format plot.
    ax.set_facecolor("w")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    plt.tight_layout()
    plt.show(block=True)


def plot_specific_bbls(bbl_list, lc, alpha=1, scatter=False, background=False,
    impath=os.environ["EXIMG"], cmap="viridis"):
    """Plot higher level classification with option to show background image.
    Args:
        bbl_list (list) - list of bbls.
        lc (obj) - LightCurve object.
        alpha (float) - alpha for plotting bbls.
        scatter (bool) - plot sources?
        background (bool) - plot example image as background?
        impath (str) - path to example image.
        cmap (str) - matplotlib colormap to use.
    """
    # -- Print status.
    tstart = start("Plotting {} bbls.".format(bbl_list))
    # -- Load bbls, and mask nans.
    # -- ...bldgclss.npy is made using bbls_to_bldgclss.
    bbls = np.load(os.path.join(lc.path_sup,  "12_3_14_bblgrid_clean.npy"))
    bbls = np.ma.array(bbls, mask=(np.isnan(bbls) | ~np.isin(bbls, bbl_list)))
    # -- Convert coords from dict to x and y lists.
    yy, xx = zip(*[lc.coords[kk] for kk, vv in lc.coords_bbls.items() if vv in bbl_list])
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(16, 8))
    if background:
        ax.imshow(adjust_img(lc, impath))
    ax.imshow(bbls, alpha=alpha, cmap=cmap, vmax=1013380040)
    if scatter:
        ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", s=3, cmap=cmap)
    # -- Plot formatting.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    ax.set_facecolor("w")
    plt.tight_layout()
    plt.show(block=True)


def plot_income_clusters(lc, clusts=7):
    """Plot residential median household income clusters on a line.
    Args:
        lc (obj) - LightCurve object.
        clusts (int) - number of clusters to plot.
    """
    # -- Print status.
    tstart = start("Plotting income clustering.")
    # -- Load dataframe of median household income.
    df   = bbl_income(lc)
    # -- Subselect for residential bbls with sources.
    res  = [kk for kk, vv in lc.coords_cls.items() if vv == 1]
    bbls = [lc.coords_bbls[idx] for idx in res]
    df   = df[np.isin(df.BBL, bbls)]
    vals = np.unique(df.Median_HH_Income[df.Median_HH_Income > 0.])
    # -- Silhouette score the median incomes.
    for nclust in range(3, 20):
        clust = cluster.KMeans(n_clusters=nclust).fit(vals.reshape(-1, 1))
        scr = silhouette_score(vals.reshape(-1, 1), clust.labels_)
        print("N Clusters: {}, Silhouette Score: {:.4f}".format(nclust, scr))
    # -- Perform clustering with nplot.
    clust = cluster.KMeans(n_clusters=clusts).fit(vals.reshape(-1, 1))
    # -- Create plot of clustered values on a line.
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.scatter(vals, np.zeros_like(vals), c=clust.labels_)
    # -- Format plot.
    ax.set_xlabel("Median Household Income")
    ax.set_yticks([])
    ax.set_title("Median Household Income Clusters (N: {})".format(nplot))
    ax.set_xlabel("Median Income")
    plt.tight_layout()
    plt.show(block=True)


# -- Writing here...
def plot_income_image(lc, res_only=True, alpha=1, scatter=False, background=False,
    impath=os.environ["EXIMG"], cmap="viridis"):
    """Plot median household income of bbls with option to show background
    image and scatter sources.
    Args:
        lc (obj) - LightCurve object.
        res_only (bool) - only show res bbls.
        alpha (float) - alpha for plotting median household income.
        scatter (bool) - plot sources?
        background (bool) - plot background image?
        impath (str) - path to example image.
        cmap (str) - matplotlib cmap for plotting median household income.
    """
    # -- Print status.
    tstart = start("Plotting bbls income in scene.")
    # -- Load income df.
    df = bbl_income(lc).set_index("BBL")
    # -- Load BBL map.
    bbls = np.load(os.path.join(lc.path_sup,  "12_3_14_bblgrid_clean.npy"))
    res  = np.load(os.path.join(lc.path_sup, "12_3_14_bblgrid_clean_bldgclss.npy"))
    # -- Map bbls to income.
    tstart = start("Mapping bbls to median household income.")
    income = [df.Median_HH_Income.get(bbl, np.nan) for bbl in bbls.ravel()]
    finish(tstart)
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(16, 8))
    if background:
        ax.imshow(adjust_img(lc, impath))
    if res_only:
        res_inc = np.ma.array(np.array(income).reshape(*bbls.shape), mask=(res != 1))
        im = ax.imshow(res_inc, alpha=alpha, cmap=cmap)
    else:
        im = ax.imshow(np.array(income).reshape(*bbls.shape), alpha=alpha, cmap=cmap)
    if scatter:
        yy, xx = zip(*lc.coords.values())
        ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", s=3)
    # -- Plot formatting.
    ax.set_title("Median Household Income ($USD)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    ax.set_facecolor("w")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_alpha(1)
    cbar.draw_all()
    plt.tight_layout()
    plt.show(block=True)


def plot_income_bigoffs_hist(lc):
    """3-panel figure showing the histograms of bigoff time split by median
    household income.
    Args:
        lc (obj) - LightCurve object with loaded bigoffs df.
    """
    # -- Load income df.
    df = bbl_income(lc).set_index("BBL")
    # -- Select bbls with sources and corresponding median household income.
    bbls = np.unique(lc.coords_bbls.values())
    bbls = df.index[df.index.isin(bbls)]
    df   = df.loc[bbls] # -- Subselect df.
    # -- Subselect bbls from by income.
    bbls_0k30k  = df[df.Median_HH_Income < 30000].index
    bbls_30k90k = df[(df.Median_HH_Income > 30000) & (df.Median_HH_Income < 90000)].index
    bbls_90k    = df[df.Median_HH_Income > 90000].index
    # -- Subselect bigoffs by the bbls.
    crds_0k30k  = np.array([kk for kk, vv in lc.coords_bbls.items()
                            if lc.coords_bbls[kk] in bbls_0k30k]) - 1
    crds_30k90k = np.array([kk for kk, vv in lc.coords_bbls.items()
                            if lc.coords_bbls[kk] in bbls_30k90k]) - 1
    crds_90k    = np.array([kk for kk, vv in lc.coords_bbls.items()
                            if lc.coords_bbls[kk] in bbls_30k90k]) - 1
    # -- Only consider residential sources.
    res_crds = [kk for kk, vv in lc.coords_cls.items() if vv == 1]
    crds_0k30k = crds_0k30k[np.isin(crds_0k30k, res_crds)]
    crds_30k90k = crds_30k90k[np.isin(crds_30k90k, res_crds)]
    crds_90k = crds_90k[np.isin(crds_90k, res_crds)]
    # -- Pull relevant bigoffs.
    boffs_0k30k  = filter(lambda x: ~np.isnan(x), lc.lc_bigoffs[crds_0k30k].values.flatten())
    boffs_30k90k = filter(lambda x: ~np.isnan(x), lc.lc_bigoffs[crds_30k90k].values.flatten())
    boffs_90k    = filter(lambda x: ~np.isnan(x), lc.lc_bigoffs[crds_90k].values.flatten())
    # -- Plot
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    ax1.hist(boffs_0k30k, 31, normed=True)
    ax2.hist(boffs_30k90k, 31, normed=True)
    ax3.hist(boffs_90k, 31, normed=True)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Hour")
        ax.set_xlim(0, 3000)
        ax.set_yticks([])
        ax.set_xticks(np.array(range(9)) * 360)
        ax.set_xticklabels(["{}".format(ii % 24) for ii in range(21, 30)])
    ax1.set_ylabel("Relative Counts")
    ax1.set_title("Median Income < $30,000 Bigoffs", fontsize=12)
    ax2.set_title("\$30,000 <= Median Income\n< $90,000 Bigoffs", fontsize=12)
    ax3.set_title("\$90,000 <= Median Income Bigoffs", fontsize=12)
    plt.tight_layout(w_pad=0.05)
    plt.show(block=True)


def plot_income_bigoffs_boxplot(lc):
    """Plot boxplots comparing bigoffs for different income brackets.
    Args:
        lc (obj) - LightCurve object.
    """
    # -- Load income df.
    df = bbl_income(lc).set_index("BBL")
    # -- Select bbls with sources and corresponding median household income.
    bbls = np.unique(lc.coords_bbls.values())
    bbls = df.index[df.index.isin(bbls)]
    df   = df.loc[bbls] # -- Subselect df.
    # -- Subselect bbls from by income.
    bbls_0k30k  = df[df.Median_HH_Income < 30000].index
    bbls_30k90k = df[(df.Median_HH_Income > 30000) & (df.Median_HH_Income < 90000)].index
    bbls_90k    = df[df.Median_HH_Income > 90000].index
    # -- Subselect bigoffs by the bbls.
    crds_0k30k  = np.array([kk for kk, vv in lc.coords_bbls.items()
                            if lc.coords_bbls[kk] in bbls_0k30k]) - 1
    crds_30k90k = np.array([kk for kk, vv in lc.coords_bbls.items()
                            if lc.coords_bbls[kk] in bbls_30k90k]) - 1
    crds_90k    = np.array([kk for kk, vv in lc.coords_bbls.items()
                            if lc.coords_bbls[kk] in bbls_30k90k]) - 1
    # -- Only consider residential sources.
    res_crds = [kk for kk, vv in lc.coords_cls.items() if vv == 1]
    crds_0k30k = crds_0k30k[np.isin(crds_0k30k, res_crds)]
    crds_30k90k = crds_30k90k[np.isin(crds_30k90k, res_crds)]
    crds_90k = crds_90k[np.isin(crds_90k, res_crds)]
    # -- Pull relevant bigoffs.
    boffs_0k30k  = filter(lambda x: ~np.isnan(x), lc.lc_bigoffs[crds_0k30k].values.flatten())
    boffs_30k90k = filter(lambda x: ~np.isnan(x), lc.lc_bigoffs[crds_30k90k].values.flatten())
    boffs_90k    = filter(lambda x: ~np.isnan(x), lc.lc_bigoffs[crds_90k].values.flatten())
    # -- Plot.
    labs = ["< $30,000", "\$30,000 - $90,000", "> $90,000"]
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.boxplot([boffs_0k30k, boffs_30k90k, boffs_90k], labels=labs, vert=False,
        positions=[0.0, 0.2, 0.4])
    # -- Plot format.
    ax.set_ylim(-0.1, 0.5)
    ax.set_xticks(np.array(range(9)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 30)])
    ax.set_xlabel("Time")
    ax.set_title("Bigoffs by Median Household Income")
    plt.tight_layout()
    plt.show(block=True)


def plot_scatter_bigoffs_income(lc):
    """Scatter plot of median household income and median bigoff time for all
    residential sources.
    Args:
        lc (obj) - LightCurve object.
    """
    # -- Load income df.
    df = bbl_income(lc).set_index("BBL")
    # -- Pull residential idx and bbls.
    crds = np.array([kk for kk, vv in lc.coords_cls.items() if vv == 1])
    bbls = [lc.coords_bbls[crd] for crd in crds]
    # -- Pull median residential bigoffs times.
    xx = np.nanmedian(lc.lc_bigoffs[crds - 1], axis=0)
    yy = df.loc[bbls].Median_HH_Income
    xx, yy = zip(*filter(lambda x: ~np.isnan(x[1]), zip(xx, yy)))
    # -- Best fit.
    mm, bb, r2, _, _ = linregress(xx, yy)
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(xx, yy, label="Residential Sources")
    ax.plot([0, 8 * 360], [0 * mm + bb, 8 * 360 * mm + bb], c="k", label="Linear Regression")
    # -- Format plot.
    ax.text(7 * 360, 170000, "y = {:.2f}x + {:.2f}".format(mm, bb), ha="right")
    ax.text(7 * 360, 160000, "R2: {:.2f}".format(r2**2), ha="right")
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
    ax.set_xlim(0, 7 * 360)
    ax.set_xlabel("Bigoff Time")
    ax.set_ylabel("Median Househould Income")
    ax.set_title("Household Income For Residential Sources by Median Bigoff Time")
    ax.legend()
    plt.show()


def plot_mean_lightcurve_by_class(lc):
    """Plot a mean light curve for each higher level building classification.
    Args:
        lc (obj) - LightCurves object.
    """
    # -- Pull source indices for all higher level classifications.
    res, _ = zip(*filter(lambda x: x[1] == 1, lc.coords_cls.items()))
    com, _ = zip(*filter(lambda x: x[1] == 2, lc.coords_cls.items()))
    mix, _ = zip(*filter(lambda x: x[1] == 3, lc.coords_cls.items()))
    ind, _ = zip(*filter(lambda x: x[1] == 4, lc.coords_cls.items()))
    mis, _ = zip(*filter(lambda x: x[1] == 5, lc.coords_cls.items()))
    # -- Pull mean light curve for each class.
    res_ts = lc.lcs[:, np.array(res) - 1].mean(axis=1)
    com_ts = lc.lcs[:, np.array(com) - 1].mean(axis=1)
    mix_ts = lc.lcs[:, np.array(mix) - 1].mean(axis=1)
    ind_ts = lc.lcs[:, np.array(ind) - 1].mean(axis=1)
    mis_ts = lc.lcs[:, np.array(mis) - 1].mean(axis=1)
    all_ts = lc.lcs.mean(axis=1)
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res_ts, label="Residential (N: {})".format(len(res)))
    ax.plot(com_ts, label="Commercial (N: {})".format(len(com)))
    ax.plot(mix_ts, label="Mixed Use (N: {})".format(len(mix)))
    ax.plot(ind_ts, label="Industrial Use (N: {})".format(len(ind)))
    ax.plot(mis_ts, label="Misc. (N: {})".format(len(mix)))
    # -- Format plot.
    ax.set_xlim(0, len(res_ts))
    ax.set_xlabel("Timesteps")
    ax.set_yticklabels([])
    ax.set_ylabel("Mean Intensity [arb units]")
    ax.set_title("Mean Lightcurve By Building Classification ({})".format(lc.night.date()))
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
    ax.legend()
    plt.show()


def plot_all_lightcurves(lc):
    """Plot all lightcurves for a given night as loaded in lc, split across 4
    panels: residential, commerical, mixed use, and misc.
    Args:
        lc (obj) - LightCurves object.
    """
    # -- Pull source indices for all higher level classifications.
    res = np.array([kk for kk, vv in lc.coords_cls.items() if vv == 1])
    com = np.array([kk for kk, vv in lc.coords_cls.items() if vv == 2])
    mix = np.array([kk for kk, vv in lc.coords_cls.items() if vv == 3])
    ind = np.array([kk for kk, vv in lc.coords_cls.items() if vv == 4])
    mis = np.array([kk for kk, vv in lc.coords_cls.items() if vv == 5])
    # -- Min max all lcs.
    data = MinMaxScaler().fit_transform(lc.lcs).T
    # -- Create plot.
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 12),
        sharey=True, sharex=True)
    for ii in data[res - 1]:
        ax1.plot(ii, c="k", alpha=0.01, lw=0.05)
    for ii in data[com - 1]:
        ax2.plot(ii, c="k", alpha=0.01, lw=0.05)
    for ii in data[mix - 1]:
        ax3.plot(ii, c="k", alpha=0.01, lw=0.05)
    for ii in data[mis - 1]:
        ax4.plot(ii, c="k", alpha=0.01, lw=0.05)
    # -- Format plot.
    for ax in fig.axes:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(ii))
        ax.set_yticks([])
        ax.set_xticks([])
    for ax in [ax3, ax4]:
        ax.set_xticks(np.array(range(8)) * 360)
        ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
        ax.set_xlabel("Time")
    ax1.set_title("Residential Sources ({})".format(lc.night.date()))
    ax2.set_title("Commercial Sources ({})".format(lc.night.date()))
    ax3.set_title("Mixed Sources ({})".format(lc.night.date()))
    ax4.set_title("Misc. Sources ({})".format(lc.night.date()))
    plt.tight_layout()
    plt.show()


def plot_preprocessing(lc, idx=3004):
    """Plot an example of the preprocessing each lightcurve undergoes.
    Args:
        lc (obj) - LightCurve object.
        idx (int) - source index to utilize.
    """
    # -- Preprocess loaded lightcurves.
    data   = MinMaxScaler().fit_transform(lc.lcs)
    gfdata = MinMaxScaler().fit_transform(gf(lc.lcs, (30, 0)))
    med    = gf(np.median(gfdata, axis=1), 30)
    mev = np.vstack([med, np.ones(med.shape)]).T
    fit = np.matmul(np.linalg.inv(np.matmul(mev.T, mev)),
                    np.matmul(mev.T, gfdata))
    model = med * fit[0].reshape(-1, 1) + fit[1].reshape(-1, 1)
    dtrend = gfdata.T - model
    # -- Plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data[:, idx], label="Original LC", alpha=0.5)
    ax.plot(gfdata[:, idx], label="GF LC", c="k", alpha=0.5)
    ax.plot(med, label="Median (All LCs)", ls="dashed", c="g")
    ax.plot(model[idx], label="Fitted Median", ls="dotted", c="g")
    ax.plot(dtrend[idx], label="Detrended LC", c="k")
    # -- Format plot.
    ax.set_xlim(0, lc.lcs.shape[0])
    ax.set_yticks([])
    ax.set_ylabel("Intensity [Arb. Units]")
    ax.set_xlabel("Timesteps")
    ax.set_title("Example Preprocessing (Src: {}, {})".format(idx, lc.night.date()))
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
    ax.set_xlabel("Time")
    ax.legend()
    plt.show()


def plot_match(img, ref, match, figsize=(6, 8)):
    """Plot image, reference, and resulting matches for histogram matching.
    Args:
        img (array) - RGB image.
        ref (array) - RGB reference image.
        match (array) - RGB histogram matched image
    """
    # -- Create figure.
    fig, [r1, r2, r3, r4] = plt.subplots(nrows=4, ncols=3, figsize=figsize)
    # -- Plot all reference and image channels.
    for ii, (ref_ax, img_ax, new_ax) in enumerate([r1, r2, r3, r4]):
        if ii < 3:
            ref_ax.imshow(ref[:, :, ii], cmap="gray")
            img_ax.imshow(img[:, :, ii], cmap="gray")
            new_ax.imshow(match[:, :, ii], cmap="gray")
        else:
            ref_ax.imshow(ref)
            img_ax.imshow(img)
            new_ax.imshow(match)
    # -- Axes labels.
    for ax, label in zip([r1[0], r2[0], r3[0], r4[0]], ["R", "G", "B", "Color"]):
        ax.set_ylabel(label)
    for ax, label in zip(r1, ["Reference", "Image", "Match"]):
        ax.set_title(label)
    # -- Formatting.
    for ii in fig.axes:
        ii.set_xticks([])
        ii.set_yticks([])
    plt.tight_layout(h_pad=0.0001, w_pad=0.1)
    plt.show(block=True)


def all_feature_imporances(path, ordinal=False, save=False):
    """Plot the feature importance of all trained classifier in either real
    values or ranked order.
    Args:
        clf (obj) - sklearn object with .feature_importances_ attribute.
        ordinal (bool) - Plot real values or rank.
        save (bool) - save the plot?
    """
    # -- Warning.
    start("THIS WILL TAKE >10 MIN TO LOAD ALL CLASSIFIERS")
    # -- Pull fnames for all classifiers.
    fnames = filter(lambda x: x.endswith(".pkl"), os.listdir(path))
    fimp = []
    # -- For each classifier load and save the feature importances.
    for fname in sorted(fnames):
        clf = joblib.load(os.path.join(path, fname))
        fimp.append(clf.feature_importances_)
        start("Loading {}".format(fname), True)
    # -- Stack feature importances.
    fimp = np.vstack(fimp)
    # -- Create plot.
    fig, ax = plt.subplots(figsize=(6, 3))
    if ordinal: # -- If ordinal show ranked feature importance.
        for ii in fimp:
            sc = ax.scatter(ii.argsort(), range(len(ii)), s=1, c="k", alpha=0.2)
        handles = [sc]
        labels  = ["Sample Feature Importance"]
    else: # -- Show absolute feature importance.
        for ii in fimp:
            sc = ax.scatter(range(len(ii)), ii, s=1, c="k", alpha=0.2)
        ll = ax.plot(fimp.mean(0))
        handles = [sc, ax.get_lines()[0]]
        labels  = ["Sample Feature Importance", "Mean Feature Importance"]
    # -- Format plot.
        ax.set_ylim(fimp.min(), fimp.max())
    ax.set_xlim(0, len(fimp[0]))
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["{}:00".format(ii % 24) for ii in range(21, 29)])
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Relative Feature Importance", fontsize=10)
    ax.legend(handles=handles, labels=labels)
    if save:
        fig.savefig("FeatureImportances.png", bbox_inches="tight")
    plt.show()


def bbls_to_bldgclass(bblmap):
    """Map bbls to higher level building classification.
    Args:
        bblmap (array) - 2d array of bbls.
    Returns:
        bbls (array) - 2d array of buidling classifications.
    """
    # -- Map bbl to building classification.
    bbls = np.array([lc.dd_bbl_bldgclss.get(val, np.nan)
                     for val in bblmap.ravel()]).reshape(*bblmap.shape)
    # -- Map building classification to higher level.
    bbls = np.array([np.nan if val == "nan" else lc.dd_bldgclss \
                     .get([k for k in lc.dd_bldgclss.keys()
                           if val.startswith(k)][0], np.nan)
                           for val in bbls.ravel()]).reshape(*bbls.shape)
    return bbls


def Fig1Scene(lc, save=False):
    """"""
    lc.loadnight(lc.meta.index[4])
    # --
    plt.style.use("ggplot")
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, figsize=(7, 10))
    # --
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["royalblue", "orange",])
    # -- Plot example image.
    img = adjust_img(lc, os.environ["EXIMG"])
    ax1.imshow(img)
    # -- Load BBL Map.
    bbls = np.load(os.path.join(lc.path_sup, "12_3_14_bblgrid_clean_bldgclss.npy"))
    bbls = np.ma.array(bbls, mask=np.isnan(bbls)) == 1
    ax2.imshow(img)
    ax2.imshow(bbls, cmap=cmap, alpha=0.3)
    # -- Plot with sources.
    ax3.imshow(img)
    ax3.imshow(bbls, cmap=cmap, alpha=0.2)
    ryy, rxx = zip(*[lc.coords[k] for k, v in lc.coords_cls.items() if v == 1])
    nyy, nxx = zip(*[lc.coords[k] for k, v in lc.coords_cls.items() if v != 1])
    ax3.scatter(np.array(rxx) - 20, np.array(ryy) - 20, marker="s", s=2,
                c="orange", label="Residential Source")
    ax3.scatter(np.array(nxx) - 20, np.array(nyy) - 20, marker="s", s=2,
                c="royalblue", label="Commercial Source")
    ax3.set_xlim(0, bbls.shape[1])
    # -- Formatting
    for ii, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.01, 0.97, "abc"[ii] + ")", ha="left", va="top", color="w",
                transform=ax.transAxes, family="helvetica", fontsize=16)
    # -- Legend.
    bp = mpatches.Patch(color="royalblue", label="Non-Residential Building", alpha=0.3)
    op = mpatches.Patch(color="orange", label="Residential Building", alpha=0.3)
    bm = plt.scatter([], [], marker="s", c="royalblue", label="Non-Residential Source")
    om = plt.scatter([], [], marker="s", c="orange", label="Residential Source")
    leg = ax3.legend(handles=[bm, om, bp, op], loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, -0.01), fontsize=12)
    plt.tight_layout(h_pad=0.1)
    plt.setp(leg.texts, family="helvetica")
    if save:
        fig.savefig("Scene.png", bbox_inches="tight")
    plt.show()


def Fig2SourceAcc(path, fname_start, save=False):
    """"""
    # --
    regex  = re.compile(fname_start + "(\d+).npy")
    fnames = sorted(filter(regex.search, os.listdir(path)))
    # --
    acc, res, nrs = [], [], []
    for fname in fnames:
        pred, test = np.load(os.path.join(path, fname))
        vals = zip(pred, test)
        acc.append(accuracy_score(*zip(*vals)))
        res.append(accuracy_score(*zip(*filter(lambda x: x[1] == 1, vals))))
        nrs.append(accuracy_score(*zip(*filter(lambda x: x[1] == 0, vals))))
    # --
    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(8, 2.66))
    plt.rcParams["font.family"] = "helvetica"
    for ii, (ax, vals) in enumerate(zip(fig.axes, [acc, res, nrs])):
        ax.text(0.01, 0.97, "abc"[ii] + ")", ha="left", va="top", color="k",
                alpha=0.5, transform=ax.transAxes, fontsize=16)
        med = np.median(np.array(vals))
        ax.axvline(med, ls="dashed", c="k", lw=1, label="Median")
        ax.hist(vals, np.array(range(51)) / 50., normed=True)
        ax.set_xlim(0.4, 1)
        ax.set_xticks((np.array(range(7)) + 4) / 10.)
        ax.set_xticklabels((np.array(range(7)) + 4) * 10, fontsize=10)
        ax.set_yticks([])
    ax1.set_ylabel("Relative Count", fontsize=12)
    ax2.set_xlabel("Accuracy (%)", fontsize=12)
    ax3.legend(prop=mpl.font_manager.FontProperties(size=12), frameon=False,
        bbox_to_anchor=(1.05, 1.05), loc="upper right")
    plt.tight_layout(w_pad=0.05)
    if save:
        fig.savefig("SourceAcc.png", bbox_inches="tight")
    plt.show()


def Fig3VoteAcc(path, fname_start, rsplit=0.5, save=False):
    """"""
    # --
    regex  = re.compile(fname_start + "(\d+).npy")
    fnames = sorted(filter(regex.search, os.listdir(path)))
    # --
    acc, res, nrs = [], [], []
    for fname in fnames:
        pred, test = np.load(os.path.join(path, fname))
        _, vacc, vnracc, vracc = votescore(pred, test, rsplit=rsplit)
        acc.append(vacc)
        res.append(vracc)
        nrs.append(vnracc)
    # --
    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(8, 2.66))
    plt.rcParams["font.family"] = "helvetica"
    for ii, (ax, vals) in enumerate(zip(fig.axes, [acc, res, nrs])):
        ax.text(0.01, 0.97, "abc"[ii] + ")", ha="left", va="top", color="k",
                alpha=0.5, transform=ax.transAxes, fontsize=16)
        med = np.median(np.array(vals))
        ax.axvline(med, ls="dashed", c="k", lw=1, label="Median")
        ax.hist(vals, np.array(range(51)) / .5, normed=True)
        ax.set_xlim(35, 100)
        ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
        ax.set_xticklabels([40, 50, 60, 70, 80, 90, 100], fontsize=10)
        ax.set_yticks([])
    ax1.set_ylabel("Relative Count", fontsize=12)
    ax2.set_xlabel("Accuracy (%)", fontsize=12)
    ax3.legend(prop=mpl.font_manager.FontProperties(size=12), frameon=False,
        bbox_to_anchor=(1.05, 1.05), loc="upper right")
    plt.tight_layout(w_pad=0.05)
    if save:
        fig.savefig("VoteAcc.png", bbox_inches="tight")
    plt.show()


def Fig4Predictions(fpath, fname_start, path, ndays=74, save=False):
    """"""
    np.random.seed(0)
    # --
    regex  = re.compile(fname_start + "(\d+).npy")
    fnames = sorted(filter(regex.search, os.listdir(fpath)))
    # --
    days, crds, lcs, ons, offs = load_data(lc, path)
    # --
    data = []
    data1 = []
    for fname in fnames:
        _ = _start("Loading {}".format(fname))
        # -- Load results and conduct vote.
        pred, test = np.load(os.path.join(fpath, fname))
        votes  = (pred.reshape(ndays, pred.size / ndays).mean(axis=0) > 0.5).astype(int)
        labels = test.reshape(ndays, test.size / ndays)[0]
        # -- Split data to match results.
        trn, trn_data, trn_labs, tst, tst_data, tst_labs = bbl_split(
            lc, crds, lcs, seed=int(fname[len(fname_start): -4]))
        idx = tst[:labels.size]
        coords = [lc.coords[ii] for ii in idx]
        # --
        df = pd.DataFrame(coords, columns=["yy", "xx"])
        df.index = idx
        df["votes"], df["labs"] = votes, labels
        df["correct"] = df["votes"] == df["labs"]
        data.append(df)
        # --
        df1 = pd.DataFrame(tst)
        df1["labs"] = tst_labs
        df1["preds"] = pred
        data1.append(df1)
    df = pd.concat(data, axis=0)
    df = df.groupby(df.index).mean()
    df1 = pd.concat(data1, axis=0)
    df1.columns = ["idx", "labs", "votes"]
    df1 = df1.groupby("idx").mean()
    coords = [lc.coords[ii] for ii in df1.index]
    df1["yy"], df1["xx"] = zip(*coords)
    df = df1
    # -- Split data for plotting.
    dfr  = df[df.labs == 1.]
    dfrc = dfr[dfr.votes > 0.5]
    dfrw = dfr[dfr.votes <= 0.5]
    dfn  = df[df.labs == 0.]
    dfnc = dfn[dfn.votes <=0.5]
    dfnw = dfn[dfn.votes > 0.5]
    # -- Create colormap.
    cmap = mcolors.LinearSegmentedColormap.from_list("",
        ["royalblue", "royalblue", "royalblue", "red", "orange", "orange", "orange"])
    # -- Create figure.
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, figsize=(7, 8),
        gridspec_kw={"height_ratios": [1, 1, 0.05]})
    # -- Load and adjust images.
    img = adjust_img(lc, os.environ["EXIMG"])
    # -- For plots with image backgrounds, set those.
    for ii, ax in enumerate([ax1, ax2]):
        ax.text(0.01, 0.97, "abc"[ii] + ")", ha="left", va="top", color="w",
                transform=ax.transAxes, family="helvetica", fontsize=16)
        ax.imshow(img.mean(-1), cmap="gist_gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img.shape[1])
    # --
    ax1.scatter(dfr.xx -20, dfr.yy - 20, c=dfr.votes, cmap=cmap, s=5, marker="s")
    cval = ax2.scatter(dfn.xx -20, dfn.yy - 20, c=dfn.votes, cmap=cmap, s=5, marker="s", vmax=1, vmin=0)
    cbar = plt.colorbar(cval, cax=ax3, orientation="horizontal")
    cbar.set_ticks([0., 0.4, 0.5, 0.6, 1.])
    cbar.ax.set_xticklabels(["Non-Res. 100%", "60%", "50%", "60%", "Res. 100%"], fontsize=12)
    tl = cbar.ax.get_xticklabels()
    tl[0].set_horizontalalignment("left")
    tl[-1].set_horizontalalignment("right")
    cbar.set_clim(0, 1)
    plt.tight_layout(h_pad=0.1)
    if save:
        fig.savefig("VotePercent.png", bbox_inches="tight")
    plt.show()
    # --
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.imshow(img.mean(-1), cmap="gist_gray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0, img.shape[1])
    ax1.scatter(dfrc.xx - 20, dfrc.yy - 20, marker="s", color="orange", s=2, label="Correctly Classified Residential Source")
    ax1.scatter(dfrw.xx - 20, dfrw.yy - 20, marker="s", color="r", s=2, label="Incorrectly Classified Residential Source")
    ax1.scatter(dfnc.xx - 20, dfnc.yy - 20, marker="s", color="c", s=2, label="Correctly Classified Non-Residential Source")
    ax1.scatter(dfnw.xx - 20, dfnw.yy - 20, marker="s", color="b", s=2, label="Incorrectly Classified Non-Residential Source")
    lgd = ax1.legend(ncol=2, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.01))
    for ii in lgd.legendHandles:
        ii._sizes = [20]
    if save:
        fig.savefig("Predictions.png", bbox_inches="tight")
    plt.show()


def Fig5VoteNights(path, fname_start, ndays=74, save=False):
    """"""
    np.random.seed(0)
    # --
    regex  = re.compile(fname_start + "(\d+).npy")
    fnames = sorted(filter(regex.search, os.listdir(path)))
    # --
    pred, test = np.load(os.path.join(path, fnames[0]))
    preds = pred.reshape(ndays, pred.size / ndays)
    # --
    labels = test.reshape(ndays, test.size / ndays)[0]
    data = []
    for ii in range(1, ndays + 1):
        arr = []
        for _ in range(100):
            idx = np.random.choice(range(ndays), ii, False)
            vote = (preds[idx].mean(axis=0) > 0.5).astype(int)
            acc = accuracy_score(vote, labels)
            arr.append(acc)
        data.append(arr)
    data = np.array(data)
    # --
    fig, ax1 = plt.subplots(figsize=(5, 3))
    for xval, yy in enumerate(data):
        if xval == 0:
            ax1.scatter(np.array([xval] * 100) + 1, yy, c="k", alpha=0.2, s=2,
                        label="Sample Accuracy")
        else:
            ax1.scatter(np.array([xval] * 100) + 1, yy, c="k", alpha=0.2, s=2)
    ax1.plot(np.array(range(ndays)) + 1, data.mean(axis=1), label="Mean Accuracy")
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_xlabel("N Observations in Voting", fontsize=12)
    ax1.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax1.set_yticklabels([50, 60, 70, 80], fontsize=10)
    ax1.set_xticklabels(np.array(range(8)) * 10, fontsize=10)
    ax1.set_xlim(0, ndays + 1)
    ax1.legend(prop=mpl.font_manager.FontProperties(size=12), frameon=False)
    plt.tight_layout()
    if save:
        fig.savefig("VotingNights.png", bbox_inches="tight")
    plt.show()


def Fig6Table(fpath, tsts, ndays=74):
    """"""
    fstarts = np.unique([fname[:-7] for fname in os.listdir(path)])
    fstarts = filter(lambda x: "full" not in x, fstarts)
    fstarts = filter(lambda x: "coords" not in x, fstarts)
    data = {}
    for fstart in sorted(fstarts):
        print(fstart)
        # --
        regex  = re.compile(fstart + "(\d+).npy")
        fnames = sorted(filter(regex.search, os.listdir(path)))
        # --
        src, vot, bld, b5, b10, b15 = [], [], [], [], [], []
        for ii, fname in enumerate(fnames):
            pred, test = np.load(os.path.join(path, fname))
            # --
            vals = zip(pred, test)
            acc = accuracy_score(*zip(*vals)) * 100
            res = accuracy_score(*zip(*filter(lambda x: x[1] == 1, vals))) * 100
            nrs = accuracy_score(*zip(*filter(lambda x: x[1] == 0, vals))) * 100
            src.append([acc, res, nrs])
            # --
            vot.append(votescore(pred, test, rsplit=rsplit, pp=False)[1:])
            # --
            df = pd.DataFrame(np.array([pred, test, tsts[ii]]).T, columns=["pred", "labs", "idx"])
            df["bbl"] = [lc.coords_bbls[idx] for idx in df.idx]
            N = df.groupby("bbl").size()
            df = df.groupby("idx").mean()
            df["pred"] = (df.pred > 0.5).astype(float)
            df = df.groupby("bbl").mean()
            df["pred"] = (df.pred > 0.5).astype(float)
            df["N"] = N / 74
            # --
            acc = accuracy_score(df.pred, df.labs) * 100
            res = accuracy_score(df[df.labs == 1.].pred, df[df.labs == 1.].labs) * 100
            nre = accuracy_score(df[df.labs == 0.].pred, df[df.labs == 0.].labs) * 100
            bld.append([acc, res, nre])
            # --
            df5 = df[df.N > 4]
            acc = accuracy_score(df5.pred, df5.labs) * 100
            res = accuracy_score(df5[df5.labs == 1.].pred, df5[df5.labs == 1.].labs) * 100
            nre = accuracy_score(df5[df5.labs == 0.].pred, df5[df5.labs == 0.].labs) * 100
            b5.append([acc, res, nre])
            # --
            df10 = df[df.N > 9]
            acc = accuracy_score(df10.pred, df10.labs) * 100
            res = accuracy_score(df10[df10.labs == 1.].pred, df10[df10.labs == 1.].labs) * 100
            nre = accuracy_score(df10[df10.labs == 0.].pred, df10[df10.labs == 0.].labs) * 100
            b10.append([acc, res, nre])
        # --
        data[fstart] = {"src": np.array(src), "vot": np.array(vot),
            "bld": np.array(bld), "b5": np.array(b5), "b10": np.array(b10)}
    # --
    for kk in fstarts:
        print("{}".format(kk))
        print("Srce -- Acc: {:.2f}, Res: {:.2f}, NRs: {:.2f}".format(
            *np.median(data[kk]["src"], axis=0)))
        print("Vote -- Acc: {:.2f}, Res: {:.2f}, NRs: {:.2f}".format(
            *np.median(data[kk]["vot"], axis=0)))
        print("Bldg -- Acc: {:.2f}, Res: {:.2f}, NRs: {:.2f}".format(
            *np.median(data[kk]["bld"], axis=0)))
        print("Bld5 -- Acc: {:.2f}, Res: {:.2f}, NRs: {:.2f}".format(
            *np.median(data[kk]["b5"], axis=0)))
        print("Bd10 -- Acc: {:.2f}, Res: {:.2f}, NRs: {:.2f}".format(
            *np.median(data[kk]["b10"], axis=0)))


def load_vals(lc, path):
    """"""
    trns = []
    tsts = []
    for ii in range(1, 101):
        trn, trn_data, trn_labs, tst, tst_data, tst_labs = bbl_split(
            lc, crds, lcs, seed=ii)
        trns.append(trn)
        tsts.append(tst)
    return [trns, tsts]
