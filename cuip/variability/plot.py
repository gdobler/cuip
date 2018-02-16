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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage.filters import gaussian_filter as gf
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
    # -- Print status.
    finish(tstart)


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
    # -- Print status.
    finish(tstart)


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
    # -- Print status.
    finish(tstart)


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
    # -- Print status.
    finish(tstart)


def plot_coords_by_idx(lc, coords):
    """"""
    yy, xx = zip(*[lc.coords[nn] for nn in coords if nn in lc.coords.keys()])
    img = adjust_img(lc, os.environ["EXIMG"])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)
    ax.imshow(np.ma.array(lc.matrix_labels, mask=~np.isin(lc.matrix_labels, coords)))
    ax.scatter(xx, yy, facecolors="none", edgecolors="r", linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_appertures(lc):
    """Plot residential/commercial split of appertures and overlay apperture
    centers.
    Args:
        lc (obj) - LightCurve object.
    """

    yy, xx = zip(*map(lambda x: lc.coords[x], lc.coords.keys()))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(lc.matrix_labels)
    ax.imshow(lc.matrix_labels[:1020], cmap="terrain")
    ax.scatter(xx, yy, c="r", s=2, marker="x")

    ax.set_title("Residential/Commercial Split of Appertures (with Centers)")
    ax.set_ylim(0, lc.matrix_labels.shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show(block=True)


def plot_imshow_lightc(lc, show=True):
    """Plot the LightCurves.
    Args:
        lc (obj) - LightCurve object.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(lc.src_lightc, cmap="gist_gray")
    ax.set_ylim(0, lc.src_lightc.shape[0])
    ax.set_title("Lightcurve for {} \nNull Sources: {}, Null %: {:.2f}" \
        .format(lc.night, len(lc.null_sources), lc.null_percent * 100))
    ax.set_xlabel("Sources")
    ax.set_ylabel("Timestep")
    ax.grid("off")
    plt.tight_layout()
    if show:
        plt.show(block=True)
    else:
        plt.savefig("./pdf/lightc_{}.png".format(lc.night))


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
    print("LIGHTCURVES: drow {}                                              " \
        .format(drow))
    print("LIGHTCURVES: dcol {}                                              " \
        .format(dcol))

    img = np.roll(read_img(img_path)[20:-20, 20:-20], (drow, dcol), (0, 1))

    return img


def plot_bbls(lc, bg_img=False, img_path=os.environ["EXAMPLEIMG"]):
    """Plot bbls, example img, and sources.
    Args:
        lc (obj) - LightCurve object.
        img_path (str) - path to example image.
    """

    # -- Get paths.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")

    # -- Load bbls, and replace 0s.
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)
    # -- Convert coords from dict to x and y lists.
    xx, yy = zip(*lc.coords.values())

    fig, ax = plt.subplots(figsize=(12, 4))
    if bg_img:
        # -- Plot image, rolling by drow and dcol.
        # -- Load image without buffer.
        img = adjust_img(lc, img_path)
        ax.imshow(img)
    ax.imshow(bbls, alpha=0.3, vmax=1013890001, cmap="flag_r")
    # -- Scatter light source coordinates taking buffer into consideration.
    uniq = np.unique(lc.coords_bbls.values())
    cmap = dict(zip(uniq, range(len(uniq))))
    colors = [cmap[key] for key in lc.coords_bbls.values()]
    ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", c=colors, s=3,
        cmap="flag_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    plt.show(block=True)


def plot_bldgclass(lc, bg_img=False, img_path=os.environ["EXAMPLEIMG"]):
    """Plot PLTUO BldgClass.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Load bbls, and replace 0s.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)

    # -- Map bbls to bldgclass number.
    bldgclass = np.array([lc.dd_bbl_n.get(bbl, -1) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1])

    bldgclass = bldgclass.astype(float)
    bldgclass[bldgclass == -1] = np.nan

    # -- Plot img.
    fig, ax = plt.subplots(figsize=(16, 8))
    if bg_img:
        img = adjust_img(lc, img_path)
        ax.imshow(img)
        im = ax.imshow(bldgclass, cmap="tab20b", alpha=0.3)
        frameon = True
    else:
        im = ax.imshow(bldgclass, cmap="tab20b")
        frameon = False
    ax.set_title("Building Class")
    ax.set_facecolor("w")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")

    # -- Create legend.
    values = np.unique(bldgclass)
    values = values[values < 100]
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[int(i)],
        label="{}".format(lc.dd_bldg_n1_r[int(i)])) for i in values[1:]]
    plt.legend(handles=patches, ncol=17, prop={"size": 7}, frameon=frameon)
    plt.tight_layout()
    plt.show(block=True)


def plot_arbclass(lc, bg_img=False, img_path=os.environ["EXAMPLEIMG"]):
    """Plot higher level building classification.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Load bbls, and replace 0s.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)

    # -- Map bbls to bldgclass.
    bldgclass = np.array([lc.dd_bbl_bldg.get(bbl, -1) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1])

    # -- Map bldgclass to arbclass num.
    arb_img = np.array([lc.dd_bldg_n2.get(bldg, -1)
        for bldg in bldgclass.ravel()]).reshape(bbls.shape[0], bbls.shape[1])

    arb_img = arb_img.astype(float)
    arb_img[arb_img == -1] = np.nan

    # -- Plot img.
    fig, ax = plt.subplots(figsize=(16, 8))
    if bg_img:
        img = adjust_img(lc, img_path)
        ax.imshow(img)
        im = ax.imshow(arb_img, cmap="tab20b", alpha=0.3)
        frameon=True
    else:
        im = ax.imshow(arb_img, cmap="tab20b")
        frameon=False

    for ii in range(1, 6):
        iicoords = [lc.coords[idx] for idx in
        filter(lambda x: lc.coords_cls[x] == ii, lc.coords_cls.keys())]
        iixx, iiyy = zip(*iicoords)
        ax.scatter(np.array(iixx)-20, np.array(iiyy)-20, marker="x", s=5)
        print("LIGHTCURVES: {} class {} sources".format(len(iixx), ii))

    ax.axhline(1050)
    ax.axhline(850)

    ax.set_title("Building Class")
    ax.set_facecolor("w")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")

    # -- Create legend.
    labs = ["Residential", "Commmercial", "Mixed Use", "Industrial", "Misc."]
    values = np.unique(arb_img)
    values = values[values > 0]
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[int(i) - 1],
        label="{}".format(labs[int(i) - 1])) for i in values]
    plt.legend(handles=patches, ncol=17, frameon=frameon)
    plt.tight_layout()
    plt.show(block=True)


def plot_specific_bbls(bbl_list, lc, bg_img=False):
    """"""

    # -- Load bbls, and replace 0s.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)

    # -- Only select relevant bbls.
    np.place(bbls, ~np.isin(bbls, center_bbls), -1)

    # -- Map bbls to bldgclass.
    bldgclass = np.array([lc.dd_bbl_bldg.get(bbl, -1) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1])

    # -- Map bldgclass to arbclass num.
    arb_img = np.array([lc.dd_bldg_n2.get(bldg, np.nan)
        for bldg in bldgclass.ravel()]).reshape(bbls.shape[0], bbls.shape[1])

    # -- Plot img.
    fig, ax = plt.subplots(figsize=(16, 8))
    if bg_img:
        img = adjust_img(lc, img_path)
        ax.imshow(img)
        im = ax.imshow(arb_img, cmap="tab20b", alpha=0.3)
        frameon=True
    else:
        im = ax.imshow(arb_img, cmap="tab20b")
        frameon=False

    ax.set_title("Building Class")
    ax.set_facecolor("w")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")

    # -- Create legend.
    labs = ["Residential", "Commmercial", "Mixed Use", "Industrial", "Misc."]
    values = np.unique(arb_img)
    values = values[values > 0]
    colors = [im.cmap(im.norm(value)) for value in range(len(labs))]
    patches = [mpatches.Patch(color=colors[int(i) - 1],
        label="{}".format(labs[int(i) - 1])) for i in values]
    plt.legend(handles=patches, ncol=17, frameon=frameon)
    plt.tight_layout()
    plt.show(block=True)


def plot_cluster_income(lc, nplot=6):
    """"""

    # -- Load residential income images.
    inc_img_path = os.path.join(lc.path_outpu, "LightCurve", "inc_img.npy")
    inc_img = np.load(inc_img_path)

    # -- What's a reasonable number of clusters?
    src_inc = {k: inc_img[v[1] - 20][v[0] - 20] for k, v in lc.coords.items()}
    vals = np.array(src_inc.values())[[ii > 0 for ii in src_inc.values()]]

    for nclust in range(3, 20):
        clust = cluster.KMeans(n_clusters=nclust).fit(vals.reshape(-1, 1))
        scr = silhouette_score(vals.reshape(-1, 1), clust.labels_)
        print("N Clusters: {}, Silhouette Score: {:.4f}".format(nclust, scr))

    clust = cluster.KMeans(n_clusters=nplot).fit(vals.reshape(-1, 1))
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.scatter(vals, np.zeros_like(vals), c=clust.labels_)
    ax.set_xlabel("Median Household Income")
    ax.set_yticks([])
    ax.set_title("Median Household Income Clusters (N: {})".format(nplot))
    ax.set_xlabel("Median Income")
    plt.tight_layout()
    plt.show()


def plot_income_bigoffs_hist(lc):
    """"""

    # -- Load residential income images.
    inc_img_path = os.path.join(lc.path_outpu, "LightCurve", "inc_img.npy")
    inc_img = np.load(inc_img_path)

    # -- Create masks for 0k-50k and >= 50k income sources.
    src_inc = {k: inc_img[v[1] - 20][v[0] - 20] for k, v in lc.coords.items()}
    mask_0k30k = [(inc > 1) & (inc < 30000)  for inc in src_inc.values()]
    mask_30k90k = [(inc >= 30000) & (inc < 90000)  for inc in src_inc.values()]
    mask_90k = [inc >= 90000 for inc in src_inc.values()]

    # -- Subselect bigoffs by masks.
    bigoff_0k30k = lc.bigoffs.loc[lc.nights, np.array(src_inc.keys())[mask_0k30k]].values
    bigoff_30k90k = lc.bigoffs.loc[lc.nights, np.array(src_inc.keys())[mask_30k90k]].values
    bigoff_90k = lc.bigoffs.loc[lc.nights, np.array(src_inc.keys())[mask_90k]].values

    # -- Select all non-nan values.
    src_0k30k = bigoff_0k30k.ravel()[~np.isnan(bigoff_0k30k.ravel())]
    src_30k90k = bigoff_30k90k.ravel()[~np.isnan(bigoff_30k90k.ravel())]
    src_90k = bigoff_90k.ravel()[~np.isnan(bigoff_90k.ravel())]

    # # -- Statistics.
    # ks_res = stat.ks_2samp(src_50k, src_0k50k)
    # resampled_0k50k = np.random.choice(src_0k50k, len(src_50k), False)
    # en_res = stat.entropy(resampled_0k50k, src_50k)

    # -- Plot
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 4))
    ax1.hist(src_0k30k, 31)
    ax2.hist(src_30k90k, 31)
    ax3.hist(src_90k, 31)

    # ax1.text(20, 20, "Entropy: {:.4f}".format(en_res), color="w", size=8)
    # ax1.text(20, 120, "KS-test (p-value): {:.4f}".format(ks_res.pvalue), color="w", size=8)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Counts")
        ax.set_xlim(0, 3000)
    ax1.set_title("Median Income < $30,000\nBigoffs", fontsize=12)
    ax2.set_title("\$30,000 <= Median Income < $90,000\nBigoffs", fontsize=12)
    ax3.set_title("\$90,000 <= Median Income\nBigoffs", fontsize=12)
    plt.tight_layout()
    plt.show(block=True)


def plot_income_bigoffs_boxplot(lc):
    """Plot boxplots comparing bigoffs for summer and winter observations.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Load residential income images.
    inc_img_path = os.path.join(lc.path_outpu, "LightCurve", "inc_img.npy")
    inc_img = np.load(inc_img_path)

    # -- Create masks for 0k-50k and >= 50k income sources.
    src_inc = {k: inc_img[v[1] - 20][v[0] - 20] for k, v in lc.coords.items()}
    src_50k_mask = [inc >= 50000 for inc in src_inc.values()]
    src_0k50k_mask = [(inc < 50000) & (inc > 1) for inc in src_inc.values()]

    # -- Subselect bigoffs by masks.
    src_50k_bigoff = lc.bigoffs.loc[lc.nights, np.array(src_inc.keys())[src_50k_mask]]
    src_0k50k_bigoff = lc.bigoffs.loc[lc.nights, np.array(src_inc.keys())[src_0k50k_mask]]

    # -- Plot.
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.boxplot([src_50k_bigoff.median(axis=1).dropna().values,
                src_0k50k_bigoff.median(axis=1).dropna().values],
        vert=False, positions=[0, 0.2],
        labels=["Median Income >= $50,000", "Median Income < $50,000"])

    ax.set_ylim(-0.1, 0.3)
    ax.set_xlabel("Timesteps")
    ax.set_title("Bigoffs by Median Household Income")

    plt.tight_layout()
    plt.show(block=True)


def plot_acs_income(lc, res=True):
    """"""

    # -- Dictionary of bbls to median household income.
    income_dict = lc.dd_bb_income

    # -- Load bbls.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")
    bbls = np.load(bbl_path)

    # -- Map bbls to bldgclass.
    bldgclass = np.array([lc.dd_bbl_bldg.get(bbl, -1) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1])

    # -- Map bldgclass to arbclass num.
    arb_img = np.array([lc.dd_bldg_n2.get(bldg, -1)
        for bldg in bldgclass.ravel()]).reshape(bbls.shape[0], bbls.shape[1])
    arb_img = arb_img.astype(float)
    arb_img[arb_img == -1] = np.nan

    # -- Map bbl to median income.
    inc_img = np.array([income_dict.get(bbl, np.nan) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1]).astype(float)

    if res:
        inc_img = inc_img * (arb_img == 1.0)

    inc_img_path = os.path.join(lc.path_outpu, "LightCurve", "inc_img.npy")
    np.save(inc_img_path, inc_img)

    arb_img[arb_img == 1.] = np.nan
    cmap = mcolors.ListedColormap(["silver"] * 4)

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(inc_img)
    ax.imshow(arb_img, cmap=cmap)
    cbar = fig.colorbar(cax, fraction=0.045, pad=0.02)
    ax.set_title("Median Household Income for Residential Buildings")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("w")
    plt.tight_layout()
    plt.show()


def plot_scatter_bigoffs_income(lc):
    """"""

    # -- Pull bigoff and income data.
    cc_inc = {k: lc.dd_bbl_income.get(v, -1) for k, v in lc.coords_bbls.items()}
    cc_bigoff = lc.bigoffs.loc[lc.nights].median(axis=0).to_dict()
    inc_bigoff = {v: cc_bigoff.get(k, -1) for k, v in cc_inc.items()}
    xx, yy = zip(*filter(lambda x: (x[0] > 0) & (x[1] > 0),
        zip(inc_bigoff.keys(), inc_bigoff.values())))

    # -- Best fit.
    mm, bb, r2, _, _ = linregress(xx, yy)

    # -- Plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(xx, yy)
    ax.plot([0, max(xx) + 10000], [bb, (max(xx) + 10000) * mm + bb])
    ax.text(1000, 0, "y = {:.5f}x + {:.5f}".format(mm, bb))
    ax.text(1000, 100, "R2: {:.5f}".format(r2**2))
    ax.set_xlim(0, max(xx) + 10000)
    ax.set_xlabel("Median Househould Income")
    ax.set_ylabel("Median Bigoff Timestep")
    ax.set_title("Bigoff Timestep v. Household Income For Residential Sources")
    plt.show()


def plot_arbclass_timeseries(lc, subsample="Percent", N=0.5, alpha=0.6):
    """"""

    # -- Use detrended min-maxed lightcurves.
    dimg = lc.src_lightc
    dimg[dimg == -9999.] = np.nan
    dimg = ((dimg - np.nanmin(dimg, axis=0)) /
            (np.nanmax(dimg, axis=0) - np.nanmin(dimg, axis=0)))

    # -- Pull source indices for all higher level classifications.
    res, _ = zip(*filter(lambda x: x[1] == 1, lc.coords_cls.items()))
    com, _ = zip(*filter(lambda x: x[1] == 2, lc.coords_cls.items()))
    mix, _ = zip(*filter(lambda x: x[1] == 3, lc.coords_cls.items()))
    ind, _ = zip(*filter(lambda x: x[1] == 4, lc.coords_cls.items()))
    mis, _ = zip(*filter(lambda x: x[1] == 5, lc.coords_cls.items()))

    if subsample == "Percent":
        res = np.random.choice(res, size=int(len(res) * N), replace=False)
        com = np.random.choice(com, size=int(len(com) * N), replace=False)
        mix = np.random.choice(mix, size=int(len(mix) * N), replace=False)
        # ind = np.random.choice(ind, size=int(len(ind) * N), replace=False)
        mis = np.random.choice(mis, size=int(len(mis) * N), replace=False)
    if subsample == "Number":
        res = np.random.choice(res, size=int(N), replace=False)
        com = np.random.choice(com, size=int(N), replace=False)
        mix = np.random.choice(mix, size=int(N), replace=False)
        # ind = np.random.choice(ind, size=int(N), replace=False)
        mis = np.random.choice(mis, size=int(N), replace=False)

    res_ts = dimg[:, np.array(res) - 1].mean(axis=1)
    com_ts = dimg[:, np.array(com) - 1].mean(axis=1)
    mix_ts = dimg[:, np.array(mix) - 1].mean(axis=1)
    # ind_ts = dimg[:, np.array(ind) - 1].mean(axis=1)
    mis_ts = dimg[:, np.array(mis) - 1].mean(axis=1)

    idx = list(res) + list(com) + list(mix) + list(mis)
    mean_ts = dimg[:, np.array(idx) - 1].mean(axis=1)

    # -- Plot.
    aa = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res_ts - mean_ts, alpha=aa, label="Residential (N: {})".format(len(res)))
    ax.plot(com_ts - mean_ts, alpha=aa, label="Commercial (N: {})".format(len(com)))
    ax.plot(mix_ts - mean_ts, alpha=aa, label="Mixed Use (N: {})".format(len(mix)))
    # ax.plot(ind_ts - mean_ts, label="Ind.") # 1 Src...
    ax.plot(mis_ts - mean_ts, alpha=aa, label="Misc. (N: {})".format(len(mix)))
    ax.set_xlim(0, dimg.shape[0])
    ax.set_xlabel("Timesteps")
    ax.set_yticklabels([])
    ax.set_ylabel("Mean I[arb](BuildingClass)  - Mean I[arb]")
    ax.set_title("Diff. in Mean Lightcurve by Building Class. from Night Mean ({})" \
        .format(lc.night))
    ax.legend()
    plt.show()


def plot_src_lightcurves(lc, data, ndates, start_idx, end_idx):
    """Successively plot lightcurves for each source between start_idx and
    end_idx from datacube.
    Args:
        lc (obj) - LightCurves object.
        data (array) - Lightcurves data cube.
        start_idx (int) - sources idx to start plotting.
        end_idx (int) - sources idx to end plotting.
    """

    for ii in range(start_idx, end_idx):
        if ii in lc.coords_cls.keys():
            dates = []
            rows = []
            for nn, df in zip(ndates, data):
                if ii in df.index:
                    dates.append(nn)
                    rows.append(np.array(df.loc[ii]))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(np.array(rows), aspect="auto")
            ax.set_title("Lightcurves For Source {} (BBL: {}, Cls: {})" \
                .format(ii, lc.coords_bbls[ii], lc.coords_cls[ii]))
            ax.set_yticks(range(len(dates)))
            ax.set_yticklabels(dates)
            ax.set_ylabel("Nights")
            ax.set_xlabel("Timesteps")
            plt.show(block=True)


def plot_datacube_curves(data, class_sources, night=0):
    """"""

    res, com, mix, mis = class_sources
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 12))

    for foo in data[:, res - 1, 1].T:
        ax1.plot(foo, c="k", alpha=0.01)
    for foo in data[:, com - 1, 1].T:
        ax2.plot(foo, c="k", alpha=0.01)
    for foo in data[:, mix - 1, 1].T:
        ax3.plot(foo, c="k", alpha=0.01)
    for foo in data[:, mis - 1, 1].T:
        ax4.plot(foo, c="k", alpha=0.01)

    ax1.set_title("Residential Sources")
    ax2.set_title("Commercial Sources")
    ax3.set_title("Mixed Sources")
    ax4.set_title("Misc. Sources")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Intensity [Arb Units]")
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 2876)
    plt.tight_layout(pad=2, h_pad=4)
    plt.show()


def plot_srcs_centrality_by_cls(data, class_sources, central="mean"):
    """"""

    res, com, mix, mis = class_sources
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 12))

    if central == "mean":
        for foo in data[:, res - 1, :].mean(axis=2).T:
            ax1.plot(foo, c="k", alpha=0.02)
        for foo in data[:, com - 1, :].mean(axis=2).T:
            ax2.plot(foo, c="k", alpha=0.02)
        for foo in data[:, mix - 1, :].mean(axis=2).T:
            ax3.plot(foo, c="k", alpha=0.02)
        for foo in data[:, mis - 1, :].mean(axis=2).T:
            ax4.plot(foo, c="k", alpha=0.02)

    elif central == "median":
        for foo in np.median(data[:, res - 1, :], axis=2).T:
            ax1.plot(foo, c="k", alpha=0.02)
        for foo in np.median(data[:, com - 1, :], axis=2).T:
            ax2.plot(foo, c="k", alpha=0.02)
        for foo in np.median(data[:, mix - 1, :], axis=2).T:
            ax3.plot(foo, c="k", alpha=0.02)
        for foo in np.median(data[:, mis - 1, :], axis=2).T:
            ax4.plot(foo, c="k", alpha=0.02)

    elif central == "std":
        for foo in np.std(data[:, res - 1, :], axis=2).T:
            ax1.plot(foo, c="k", alpha=0.02)
        for foo in np.std(data[:, com - 1, :], axis=2).T:
            ax2.plot(foo, c="k", alpha=0.02)
        for foo in np.std(data[:, mix - 1, :], axis=2).T:
            ax3.plot(foo, c="k", alpha=0.02)
        for foo in np.std(data[:, mis - 1, :], axis=2).T:
            ax4.plot(foo, c="k", alpha=0.02)

    else:
        raise("{} is not a valid central measure".format(central))

    ax1.set_title("Residential Sources")
    ax2.set_title("Commercial Sources")
    ax3.set_title("Mixed Sources")
    ax4.set_title("Misc. Sources")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Intensity [Arb Units]")
        ax.set_yticks([])
        if central == "std":
            ax.set_ylim(0, 0.5)
        else:
            ax.set_ylim(0, 1)
        ax.set_xlim(0, 2876)
    plt.tight_layout(pad=2, h_pad=4)
    plt.show()


def plot_correct_predictions(lc, pred_df, lclass, rclass, bg_img=False,
    img_path=os.environ["EXAMPLEIMG"]):
    """Correct/incorrect predictions.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Load bbls, and replace 0s.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)

    # -- Create dict of bbls:left/right.
    keys = {lc.coords_bbls[k]: 0 for k, v in lclass.items()}
    keys.update({lc.coords_bbls[k]: 1 for k, v in rclass.items()})

    # -- Map bbls to left right.
    leftright = np.array([keys.get(bbl, -1) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1])
    leftright = leftright.astype(float)
    leftright[leftright == -1] = np.nan

    # -- Create bbl:correct dict.
    pred_correct = pred_df["correct"].to_dict()

    # -- Map to correct predictions.
    correct_preds = np.array([pred_correct.get(bbl, np.nan) for bbl in bbls.ravel()]) \
        .reshape(bbls.shape[0], bbls.shape[1])

    # -- Plot img.
    fig, ax = plt.subplots(figsize=(16, 8))
    if bg_img:
        img = adjust_img(lc, img_path)
        ax.imshow(img)
        im = ax.imshow(correct_preds, cmap="cool", alpha=0.3)
        frameon=True
    else:
        im = ax.imshow(correct_preds, cmap="cool")
        frameon=False

    ax.set_title("Building Level Prediction (50% Cutoffs)")
    ax.set_facecolor("w")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")

    plt.tight_layout()
    plt.show(block=True)


def plot_preprocessing(lc):
    """"""
    data   = MinMaxScaler().fit_transform(lc.lcs)
    gfdata = MinMaxScaler().fit_transform(gf(lc.lcs, (30, 0)))
    med    = gf(np.median(gfdata, axis=1), 30)
    mev = np.vstack([med, np.ones(med.shape)]).T
    fit = np.matmul(np.linalg.inv(np.matmul(mev.T, mev)),
                    np.matmul(mev.T, gfdata))
    model = med * fit[0].reshape(-1, 1) + fit[1].reshape(-1, 1)
    dtrend = gfdata.T - model
    # -- Figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data[:, 3004], label="Original LC", alpha=0.5)
    ax.plot(gfdata[:, 3004], label="GF LC", c="k", alpha=0.5)
    ax.plot(med, label="Median (All LCs)", ls="dashed", c="g")
    ax.plot(model[3004], label="Fitted Median", ls="dotted", c="g")
    ax.plot(dtrend[3004], label="Detrended LC", c="k")
    ax.set_xlim(0, lc.lcs.shape[0])
    ax.set_yticks([])
    ax.set_ylabel("Intensity [Arb. Units]")
    ax.set_xlabel("Timesteps")
    ax.set_title("Example Preprocessing (Src: 3004, {})".format(lc.night.date()))
    ax.legend()
    plt.show()


def eval_n_estimators(lc, path):
    """"""
    days, crds, lcs, ons, offs = load_data(lc, path)
    # --
    pklpath = os.path.join(lc.path_out, "jvani_pkl")
    fnames  = os.listdir(pklpath)
    # --
    for fname in fnames:
        _ = _start(fname)
        # -- Load classifier
        clf  = joblib.load(os.path.join(pklpath, fname))
        seed = int(fname[:-4].split("_")[-1])
        # -- Train/test split keeping BBLs in the same set.
        trn, trn_data, trn_labs, tst, tst_data, tst_labs = bbl_split(
            lc, crds, lcs, seed=seed)
        # -- Whiten and append coords.
        trn_data, tst_data = preprocess(trn, trn_data, tst, tst_data)
        # --
        preds = clf.predict(tst_data)
        # --
        np.save("preds_{:03d}.npy".format(seed), np.array([tst_labs, preds]))


def summarize_n_estimators(path):
    """"""
    acc = []
    res = []
    nre = []
    bst = []
    for fname in filter(lambda x: x.startswith("preds"), os.listdir(path)):
        test, pred = np.load(os.path.join(path, fname))
        vals = zip(test, pred)
        acc.append(accuracy_score(*zip(*vals)))
        res.append(accuracy_score(*zip(*filter(lambda x: x[0] == 1, vals))))
        nre.append(accuracy_score(*zip(*filter(lambda x: x[0] == 0, vals))))
        src_mn = pred.reshape(74, pred.size / 74).mean(0)
        scr = []
        for ii in np.array(range(20)) / 20.:
            votes = (src_mn > ii).astype(int)
            v_acc = accuracy_score(test.reshape(74, test.size / 74)[0], votes)
            scr.append(v_acc)
        bst.append(scr)
    return [acc, res, nre, bst]


def plot_n_estimator_summary(acc, res, nre, bst):
    """"""
    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(12, 4), sharey=True)
    for ax, data in zip(fig.axes, [acc, res, nre]):
        ax.hist(data, 30, label="Histogram")
        ax.axvline(np.median(data), ls="dashed", label="Median", c="k", alpha=0.5)
        ax.text(np.median(data) + 0.01, 0.5, "{:.2f}%".format(np.median(data) * 100.),
                rotation=90, color="w", va="bottom")
    for ax in fig.axes:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        ax.set_xticklabels([0, 20, 40, 60, 80, 100])
        ax.set_yticks([])
    ax2.set_xlabel("Accuracy (%)")
    ax1.set_title("Overall Source Accuracy", fontsize=12)
    ax2.set_title("Residential Source Accuracy", fontsize=12)
    ax3.set_title("Non-Residential Source Accuracy", fontsize=12)
    ax3.legend()
    plt.suptitle("Accuracy Results From 308 RFs With Different Train/Test Splits")
    plt.show()

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(4*2, 5*2), sharey=True, sharex=True)
    for ii, ax in enumerate(fig.axes):
        ax.set_xlim(0, 1)
        ax.hist(bst[:, ii], 30, label="Histogram")
        ax.axvline(np.median(bst[:, ii]), ls="dashed", label="Median", c="k",
                   alpha=0.5)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.text(0.05, 55, ii / 20.)
        ax.text(0.05, 47, "{:.2f}%".format(np.median(bst[:, ii]) * 100.))
    fig.axes[2].set_title("Overall Accuracy From Source Voting")
    fig.axes[-3].set_xlabel("Accuracy (%)")
    plt.show()


def plot_tscoord_featureimportance(clf):
    """"""
    imp = clf.feature_importances_
    crd = imp[-2:]
    tms = imp[:-2]
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh([0, 1], [sum(crd), sum(tms)])
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Coordinates", "Time Series"])
    ax.set_xlabel("Cumulative RF Feature Importance")
    ax.set_title("Cumulative RF Feature Importance By Data Source")
    ax.text(0.01, 1, "N: {}".format(len(tms)), color="w", va="center")
    ax.text(0.01, 0, "N: 2", color="w", va="center")
    ax.text(sum(tms) + 0.01, 1, "{:.2f}".format(sum(tms)), color="k", alpha=0.7, va="center")
    ax.text(sum(crd) + 0.01, 0, "{:.2f}".format(sum(crd)), color="k", alpha=0.7, va="center")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(clf, ordinal=False, sort=False):
    """Plot the feature importance of the feature vector, in either real values
    or ranked order.
    Args:
        clf (obj) - sklearn object with .feature_importances_ attribute.
        ordinal (bool) - Plot real values or rank.
    """
    fimp = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(12, 4))
    if ordinal:
        ax.scatter(fimp.argsort(), range(len(fimp)), s=5)
        ax.set_yticks([])
        ax.set_ylabel("Ordinal Feature Importance")
        ax.set_title("Ordinal Feature Importance In RF Classification")
    else:
        if sort:
            ax.scatter(range(len(fimp)), fimp[fimp.argsort()], s=5)
        else:
            ax.scatter(range(len(fimp)), fimp, s=5)
        ax.set_ylabel("Feature Importance (log10)")
        ax.set_title("Feature Importance In RF Classification")
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylim(ymin=fimp.min() - 0.0001, ymax=fimp.max() + 0.1)
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels([21, 22, 23, 24, 1, 2, 3, 4])
    ax.set_xlabel("Hour")
    if sort:
        ax.set_xticks([])
        ax.set_xlabel("")
    ax.set_xlim(0, len(fimp) + 10)
    plt.show()


def bbls_to_bldgclass(bblmap):
    """"""
    # -- Convert bbl to building class.
    bbls = np.array([lc.dd_bbl_bldgclss.get(val, np.nan)
                     for val in bblmap.ravel()]).reshape(*bblmap.shape)
    bbls = np.array([np.nan if val == "nan" else
            lc.dd_bldgclss.get([k for k in lc.dd_bldgclss.keys()
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


def all_feature_imporances(path, save=False):
    """"""
    # --
    fnames = filter(lambda x: x.endswith(".pkl"), os.listdir(path))
    fimp = []
    for fname in fnames:
        clf = joblib.load(os.path.join(path, fname))
        fimp.append(clf.feature_importances_)
    fimp = np.vstack(fimp)
    # --
    fig, ax = plt.subplots(figsize=(6, 3))
    for ii in fimp:
        ax.scatter(range(len(ii)), ii, s=1, c="k", alpha=0.2)
    ax.scatter(range(len(ii)), ii, s=1, c="k", alpha=0.2, label="Sample Feature Importance")
    ax.plot(fimp.mean(0), label="Mean Feature Importance")
    ax.set_ylim(fimp.min(), fimp.max())
    ax.set_xlim(0, len(fimp[0]))
    ax.set_xticks(np.array(range(8)) * 360)
    ax.set_xticklabels(["21:00", "22:00", "23:00", "24:00" ,"1:00", "2:00", "3:00", "4:00"], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Relative Feature Importance", fontsize=10)
    ax.legend()
    if save:
        fig.savefig("FeatureImportances.png", bbox_inches="tight")
    plt.show()


def plot_match(img, ref, match, figsize=(6, 8)):
    """Plot image, reference, and resulting matches.
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
