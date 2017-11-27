from __future__ import print_function

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

plt.style.use("ggplot")


def plot_night(lc):
    """Plot image of all lightcurves on the loaded evening. Sort by bigoff time
    and overlay bigoff times.
    Args:
        lc (obj) - LightCurve object.
    """

    # -- Get data for the given night.
    dimg = lc.lightc
    dimg[dimg == -9999.] = np.nan
    dimg = (dimg - np.nanmin(dimg, axis=0)) / (np.nanmax(dimg, axis=0) - np.nanmin(dimg, axis=0))
    dimg = dimg.T
    offs = lc.bigoffs[lc.night]

    # -- Pull x and y coords (i.e., off and source),
    vals = zip(map(lambda x: x[1], offs), map(lambda x: x[0], offs))
    # -- Sort by off timestep.
    vals = sorted(vals, key=lambda x: x[0])
    # -- Sort dimg by off time.
    xx, yy = zip(*vals)
    dimg = dimg[[yy]]
    # -- Only plot off times > 0.
    valsf = filter(lambda x: x[0] > 0, vals)
    xx, yy = zip(*valsf)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(dimg, aspect="auto")
    ax.scatter(xx, np.arange(len(xx)) + (len(vals) - len(valsf)), marker="x", s=4)
    ax.set_ylim(0, dimg.shape[0])
    ax.set_xlim(0, dimg.shape[1])
    ax.set_title("Light Curves for {}".format(lc.night))
    ax.set_ylabel("Light Sources w Off")
    ax.set_xlabel("Timesteps")
    ax.grid("off")
    plt.tight_layout()
    plt.show(block=True)


def plot_lightcurve(lc, idx):
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
    bigoff_ = lc.bigoffs[lc.night][idx][1]
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


def plot_winter_summer_bigoffs_boxplot(lc, median=True):
    """Plot boxplots comparing bigoffs for summer and winter observations.
    Args:
        lc (obj) - LightCurve object.
        median (bool) - if true plot median else plot mean.
    """

    if median:
        winter = lc.df[lc.df.index.month > 9].median(axis=1).dropna()
        summer = lc.df[lc.df.index.month < 9].median(axis=1).dropna()
        centralm = "Median"
    else:
        winter = lc.df[lc.df.index.month > 9].mean(axis=1).dropna()
        summer = lc.df[lc.df.index.month < 9].mean(axis=1).dropna()
        centralm = "Mean"

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.boxplot([winter.values, summer.values],
        vert=False, labels=["Winter", "Summer"], positions=[0, 0.2])

    ax.set_ylim(-0.1, 0.3)
    ax.set_xlabel("Timesteps")
    ax.set_title("{} Bigoff Timestep for Summer and Winter".format(centralm))

    plt.tight_layout()
    plt.show()
