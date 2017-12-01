from __future__ import print_function

import cPickle
import numpy as np
import scipy.stats as stat
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


def plot_night_img(lc, show=True, res=False):
    """Plot image of all lightcurves on the loaded evening. Sort by bigoff time
    and overlay bigoff times.
    Args:
        lc (obj) - LightCurve object.
        show (bool) - Show plot or save.
    """

    # -- Get data for the given night.
    dimg = lc.src_lightc
    dimg[dimg == -9999.] = np.nan
    dimg = ((dimg - np.nanmin(dimg, axis=0)) /
            (np.nanmax(dimg, axis=0) - np.nanmin(dimg, axis=0)))
    dimg = dimg.T
    offs = lc.bigoffs.loc[lc.night].sort_values()

    if res:
        res_labs = filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())
        offs = offs.loc[res_labs].reset_index(drop=True).sort_values()
        offs.index = offs.index + 1
        dimg = dimg[np.array(res_labs) - 1]

    # -- Sort dimg by off time.
    xx, yy = offs.index.astype(int) - 1, offs.values
    dimg = dimg[[xx]]

    # -- Only plot off times > 0.
    vals = zip(offs.values, offs.index)
    valsf = filter(lambda x: x[0] > 0, vals)
    xx, yy = zip(*valsf)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(dimg, aspect="auto")
    ax.scatter(np.array(xx), np.arange(len(xx)), marker="x", s=4)
    ax.set_ylim(0, dimg.shape[0])
    ax.set_xlim(0, dimg.shape[1])
    ax.set_title("Light Curves for {}".format(lc.night))
    ax.set_ylabel("Light Sources w Off")
    ax.set_xlabel("Timesteps")
    ax.grid("off")
    plt.tight_layout()

    if show:
        plt.show(block=True)
    else:
        plt.savefig("./pdf/night_{}.png".format(lc.night))


def plot_lightcurve_line(lc, idx):
    """Plot single lightcurve from the loaded evening and overlay with all ons,
    offs, and the bigoff.
    Args:
        lc (obj) - LightCurve object.
        idx (int) - source index to plot.
    """

    # -- Get data for the given night.
    dimg = lc.src_lightc.T
    dimg[dimg == -9999.] = np.nan
    offs = lc.src_offs.T
    ons = lc.src_ons.T
    bigoff_ = lc.bigoffs.loc[lc.night].loc[idx + 1]
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


def plot_winter_summer_bigoffs_boxplot(lc, res=True):
    """Plot boxplots comparing bigoffs for summer and winter observations.
    Args:
        lc (obj) - LightCurve object.
    """

    bidx = lc.bigoffs.index
    winter = lc.bigoffs[(bidx.month > 9) & (bidx.dayofweek < 5)]
    summer = lc.bigoffs[(bidx.month < 9) & (bidx.dayofweek < 5)]

    if res:
        res_labs = filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())
        # res_labs = filter(lambda x: lc.coords[x][1] > 1020, lc.coords.keys())
        winter = winter[res_labs].median(axis=1).dropna()
        summer = summer[res_labs].median(axis=1).dropna()
        title = "Median Weekday {} Bigoff Timestep for Summer and Winter" \
            .format("Residential")
    else:
        winter = winter.median(axis=1).dropna()
        summer = summer.median(axis=1).dropna()
        title = "Median Weekday Bigoff Timestep for Summer and Winter"

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.boxplot([winter.values, summer.values],vert=False, positions=[0, 0.2],
        labels=["Winter", "Summer"])

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
    ax.imshow(lc.mat_labs)
    ax.imshow(lc.mat_labs[:1020], cmap="terrain")
    ax.scatter(xx, yy, c="r", s=2, marker="x")

    ax.set_title("Residential/Commercial Split of Appertures (with Centers)")
    ax.set_ylim(0, lc.mat_labs.shape[0])
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
    fnum = lc.meta.loc[pd.datetime(yyyy, mm, dd).date()]["fname"]
    reg = pd.read_csv(os.path.join(lc.path_regis, "register_{}.csv".format(fnum)))
    img_reg = reg[reg.fname == os.path.basename(img_path)]
    drow = img_reg.drow.values.round().astype(int)[0]
    dcol = img_reg.dcol.values.round().astype(int)[0]

    img = np.roll(read_img(img_path)[20:-20, 20:-20], (drow, dcol), (0, 1))

    return img


def plot_bbls(lc, img_path=os.environ["EXAMPLEIMG"]):
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
    # -- Load image without buffer.
    img = adjust_img(lc, img_path)

    fig, ax = plt.subplots(figsize=(12, 4))
    # -- Plot image, rolling by drow and dcol.
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

    # -- Path to bldgclass.np
    bldgclass_path = os.path.join(lc.path_outpu, "bldgclass.npy")

    # -- Load bbls, and replace 0s.
    bbl_path = os.path.join(lc.path_suppl, "12_3_14_bblgrid_clean.npy")
    bbls = np.load(bbl_path)
    np.place(bbls, bbls == 0,  np.min(bbls[np.nonzero(bbls)]) - 100)

    # -- Map BBL to BldgClass.
    try:
        bldgclass = np.load(bldgclass_path)
    except:
        print("bldgclass.npy does not exist!")
        bldgclass = np.array([lc.dd_bbl_n[bbl] if bbl in lc.dd_bbl_n.keys() else -1
            for bbl in bbls.ravel()]).reshape(bbls.shape[0], bbls.shape[1])

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

    # -- Define required paths.
    bldgclass_path = os.path.join(lc.path_outpu, "bldgclass.npy")
    arbimg_path = os.path.join(lc.path_outpu, "arbimg.npy")

    narbclass = {lc.dd_bldg_n2[k]: v for k, v in lc.dd_bldg_n2.items()}
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


def plot_winter_summer_hist(lc, res=True):
    """"""

    bidx = lc.bigoffs.index
    winter = lc.bigoffs[(bidx.month > 9) & (bidx.dayofweek < 5)]
    summer = lc.bigoffs[(bidx.month < 9) & (bidx.dayofweek < 5)]

    if res:
        res_labs = filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())
        winter = winter[res_labs]
        summer = summer[res_labs]

    wbigoffs = winter.values.ravel()[~np.isnan(winter.values.ravel())]
    sbigoffs = summer.values.ravel()[~np.isnan(summer.values.ravel())]

    ks_res = stat.ks_2samp(wbigoffs, sbigoffs)
    resampled_wbig = np.random.choice(wbigoffs, len(sbigoffs), False)
    en_res = stat.entropy(resampled_wbig, sbigoffs)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
    ax1.hist(wbigoffs, 31)
    ax2.hist(sbigoffs, 31)

    ax2.text(0, 0, "Entropy: {}".format(en_res), color="w")
    ax2.text(0, 100, "KS-test (p-value): {}".format(ks_res.pvalue), color="w")

    ax1.set_title("Winter Bigoffs")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Counts")
        ax.set_xlim(0, 3000)
    ax2.set_title("Summer Bigoffs")
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    plot_night_img(lc)
    plot_lightcurve_line(lc, 136)
    plot_winter_summer_bigoffs_boxplot(lc, res=True)
    plot_winter_summer_hist(lc)
    plot_appertures(lc)
    plot_bbls(lc)
    plot_bldgclass(lc)
    plot_arbclass(lc)
