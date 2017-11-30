from __future__ import print_function

import cPickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches
import scipy.ndimage.measurements as ndm

plt.style.use("ggplot")


def read_img(file_path):
    """Read .raw or .png image file.
    Args:
        file_path (str) - path to raw image.
    Returns:
        img (array) - image as np array."""

    if file_path.endswith(".raw"):
        img = np.fromfile(file_path, dtype=np.uint8) \
                .reshape(2160, 4096, 3)[...,::-1]

    if file_path.endswith(".png"):
        img = imread(file_path)

    return img


def plot_night_img(lc):
    """Plot image of all lightcurves on the loaded evening. Sort by bigoff time
    and overlay bigoff times.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Get data for the given night.
    dimg = lc.lightc
    dimg[dimg == -9999.] = np.nan
    dimg = ((dimg - np.nanmin(dimg, axis=0)) /
            (np.nanmax(dimg, axis=0) - np.nanmin(dimg, axis=0)))
    dimg = dimg.T
    offs = lc.bigoffs.loc[lc.night].sort_values()

    # -- Sort dimg by off time.
    vals = zip(offs.values, offs.index)
    xx, yy = zip(*vals)
    dimg = dimg[[yy]]

    # -- Only plot off times > 0.
    valsf = filter(lambda x: x[0] > 0, vals)
    xx, yy = zip(*valsf)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(dimg, aspect="auto")
    ax.scatter(xx, np.arange(len(xx)), marker="x", s=4)
    ax.set_ylim(0, dimg.shape[0])
    ax.set_xlim(0, dimg.shape[1])
    ax.set_title("Light Curves for {}".format(lc.night))
    ax.set_ylabel("Light Sources w Off")
    ax.set_xlabel("Timesteps")
    ax.grid("off")
    plt.tight_layout()
    plt.show(block=True)


def plot_lightcurve_line(lc, idx):
    """Plot single lightcurve from the loaded evening and overlay with all ons,
    offs, and the bigoff.
    Args:
        lc (obj) - LightCurve object.
        idx (int) - source index to plot.
    """

    # -- Get data for the given night.
    dimg = lc.lightc.T
    dimg[dimg == -9999.] = np.nan
    offs = lc.off.T
    ons = lc.on.T
    bigoff_ = lc.bigoffs.loc[lc.night].loc[idx]
    bigoff = (bigoff_ if bigoff_ > 0 else None)

    fig, ax = plt.subplots(figsize=(9, 3))
    # -- Plot lightcurve timeseries.
    ax.plot(dimg[idx], c="k", alpha=0.6)
    # -- Plot all offs in orange.
    for ii in offs[idx].nonzero()[0]:
        ax.axvline(ii, c="orange", alpha=0.8)
    # -- Plot all ons in green.
    for ii in ons[idx].nonzero()[0]:
        ax.axvline(ii, c="g", alpha=0.8)
    # -- Plot bigoff in red.
    ax.axvline(bigoff, c="r", alpha=0.8)

    ax.set_xlim(0, len(dimg[idx]))
    ax.set_yticklabels([])
    ax.set_ylabel("Intensity [arb units]")
    ax.set_xlabel("Timestep")
    ax.set_title("Lightcurve for Source {} on {}".format(idx, lc.night))

    plt.tight_layout()
    plt.show(block=True)


def plot_winter_summer_bigoffs_boxplot(lc, median=True, res=True):
    """Plot boxplots comparing bigoffs for summer and winter observations.
    Args:
        lc (obj) - LightCurve object.
        median (bool) - if true plot median else plot mean.
    """

    if median:
        winter = lc.bigoffs[lc.bigoffs.index.month > 9]
        summer = lc.bigoffs[lc.bigoffs.index.month < 9]
        centralm = "Median"
    else:
        winter = lc.bigoffs[lc.bigoffs.index.month > 9]
        summer = lc.bigoffs[lc.bigoffs.index.month < 9]
        centralm = "Mean"

    if res:
        res_labs = filter(lambda x: lc.coords[x][1] > 1020, lc.coords.keys())
        winter = winter[res_labs].median(axis=1).dropna()
        summer = summer[res_labs].median(axis=1).dropna()
        title = "{} {} Bigoff Timestep for Summer and Winter" \
            .format(centralm, "Residential")
    else:
        winter = winter.median(axis=1).dropna()
        summer = summer.median(axis=1).dropna()
        title = "{} Bigoff Timestep for Summer and Winter".format(centralm)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.boxplot([winter.values, summer.values],
        vert=False, labels=["Winter", "Summer"], positions=[0, 0.2])

    ax.set_ylim(-0.1, 0.3)
    ax.set_xlabel("Timesteps")
    ax.set_title(title)

    plt.tight_layout()
    plt.show(block=True)


def plot_appertures(lc):
    """Plot residential/commercial split of appertures and overlay apperture
    centers.
    Args:
        lc (obj) - LightCurve object.
    """

    xx, yy = zip(*map(lambda x: lc.coords[x], lc.coords.keys()))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(lc.labs)
    ax.imshow(lc.labs[:1020], cmap="terrain")
    ax.scatter(xx, yy, c="r", s=2, marker="x")

    ax.set_title("Residential/Commercial Split of Appertures (with Centers)")
    ax.set_ylim(0, lc.labs.shape[0])
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
    ax.imshow(lc.lightc, cmap="gist_gray")
    ax.set_ylim(0, lc.lightc.shape[0])
    ax.set_title("Lightcurve for {} \nNull Sources: {}, Null %: {:.2f}" \
        .format(lc.night, len(lc.null_src), lc.null_per * 100))
    ax.set_xlabel("Sources")
    ax.set_ylabel("Timestep")
    ax.grid("off")
    plt.tight_layout()
    if show:
        plt.show(block=True)
    else:
        plt.savefig("./pdf/lightc_{}.png".format(lc.night))


def plot_bbls(lc, img_path=os.environ["EXAMPLEIMG"]):
    """Plot bbls, example img, and sources.
    Args:
        lc (obj) - LightCurve object.
        img_path (str) - path to example image.
    """

    # -- Get paths.
    bbl_path = os.path.join(lc.spath, "12_3_14_bblgrid_clean.npy")

    # -- Find drow and dcol for the example image.
    spath = img_path.split("/")
    yyyy, mm, dd = int(spath[5]), int(spath[6]), int(spath[7])
    fnum = lc.meta.loc[pd.datetime(yyyy, mm, dd).date()]["fname"]
    reg = pd.read_csv(os.path.join(lc.rpath, "register_{}.csv".format(fnum)))
    img_reg = reg[reg.fname == os.path.basename(img_path)]
    drow = img_reg.drow.values.round().astype(int)[0]
    dcol = img_reg.dcol.values.round().astype(int)[0]

    # -- Load bbls, and replace 0s.
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)
    # -- Convert coords from dict to x and y lists.
    xx, yy = zip(*lc.coords.values())
    # -- Load image without buffer.
    img = read_img(img_path)[20:-20, 20:-20]

    fig, ax = plt.subplots(figsize=(12, 4))
    # -- Plot image, rolling by drow and dcol.
    ax.imshow(np.roll(img, (drow, dcol), (0, 1)))
    ax.imshow(bbls, alpha=0.3, vmax=1013890001, cmap="flag_r")
    # -- Scatter light source coordinates taking buffer into consideration.
    uniq = np.unique(lc.coord_bbls.values())
    cmap = dict(zip(uniq, range(len(uniq))))
    colors = [cmap[key] for key in lc.coord_bbls.values()]
    ax.scatter(np.array(xx) - 20, np.array(yy) - 20, marker="x", c=colors, s=3,
        cmap="flag_r")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid("off")
    plt.show()


def plot_bldgclass(lc):
    """Plot PLTUO BldgClass.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Path to bldgclass.np
    bldgclass_path = os.path.join(lc.outp, "bldgclass.npy")

    # -- Load bbls, and replace 0s.
    bbls = np.load(lc.bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)

    # -- Map BBL to BldgClass.
    try:
        bldgclass = np.load(bldgclass_path)
    else:
        print("bldgclass.npy does not exist!")
        bldgclass = np.array([lc.bbln[bbl] if bbl in lc.bbln.keys() else -1
            for bbl in bbls.ravel()]).reshape(bbls.shape[0], bbls.shape[1])

    bldgclass = bldgclass.astype(float)
    bldgclass[bldgclass == -1] = np.nan

    # -- Plot img.
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(bldgclass, cmap="tab20b")
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
        label="{}".format(lc.classn_r[int(i)])) for i in values[1:]]
    plt.legend(handles=patches, ncol=17, prop={"size": 7}, frameon=False)
    plt.tight_layout()
    plt.show()


def create_arbclass_dict(lc):
    """Categorize BldgClass numbers.
    Args:
        lc (obj) - LightCurve object.
    """

    arbclass = {}
    res = ["B", "C", "D", "N", "R1", "R2", "R3", "R4", "S"]
    com = ["J", "K", "L", "O", "RA", "RB", "RC", "RI"]
    mix = ["RM", "RR", "RX"]
    ind = ["F"]
    mis = ["G", "H", "I", "M", "P", "Q", "T", "U", "V", "W", "Y", "Z"]
    for cc in lc.classn_r.values():
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

    return arbclass


def plot_arbclass(lc):
    """Plot higher level building classification.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Define required paths.
    bldgclass_path = os.path.join(lc.outp, "bldgclass.npy")
    arbimg_path = os.path.join(lc.outp, "arbimg.npy")

    arbclass = create_arbclass_dict(lc)
    narbclass = {lc.classn[k]: v for k, v in arbclass.items()}
    bldgclass = np.load(bldgclass_path)

    # -- Map bldgclass to arbclass.
    try:
        arb_img = np.load(arbimg_path)
    except:
        print("arbimg.py does not exist!!")
        arb_img = np.array([narbclass[cc] if cc in narbclass.keys() else -1
            for cc in bldgclass.ravel()]) \
            .reshape(bldgclass.shape[0], bldgclass.shape[1])

    arb_img = arb_img.astype(float)
    arb_img[arb_img == -1] = np.nan

    # -- Plot img.
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(arb_img, cmap="tab20b")
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
    plt.legend(handles=patches, ncol=17, frameon=False)
    plt.tight_layout()
    plt.show()
