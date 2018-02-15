from __future__ import print_function

import os
import sys
import time
import cPickle
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def empirical_bootstrap(lc, iters):
    """Emprically bootstrap winter and summer bigoff times.
    Args:
        lc (obj) - LightCurves object.
        iters (int) - Number of draws.
    Returns:
        results (dict) - mean, median, and std for each sample.
    """

    tstart = time.time()
    print("LIGHTCURVES: Bootstrapping bigoffs...                              ")
    sys.stdout.flush()

    results = {"summer": [], "winter": []}

    # -- Reshape to 1D array and select all real number bigoffs.
    sum_values = lc.bigoffs_sum.values.ravel()
    win_values = lc.bigoffs_win.values.ravel()
    bigoffs_sum = sum_values[~np.isnan(sum_values)].astype(int)
    bigoffs_win = win_values[~np.isnan(win_values)].astype(int)

    # -- Set dist to fit.
    dist = scipy.stats.burr

    np.random.seed(0)
    for ii in range(iters):
        # -- Create summer/winter bootstrapped sample.
        boot_summer = np.random.choice(bigoffs_sum, len(bigoffs_sum))
        boot_winter = np.random.choice(bigoffs_win, len(bigoffs_win))

        # -- Calculate mean, median, and std.
        for key, sample in [("summer", boot_summer), ("winter", boot_winter)]:
            param = dist.fit(sample)
            results[key].append([sample.mean(), np.median(sample), sample.std(),
                                 param, dist.mean(*param), dist.median(*param),
                                 dist.std(*param)])

        # -- Print status:
        print( """LIGHTCURVES: Bootstrap sample ({}/{})                    """ \
           .format(ii+1, iters), end="\r")
        sys.stdout.flush()

    # -- Write results to file.
    folder = os.path.join(lc.path_outpu, "bootstrapping")
    fname = os.path.join(folder, "empirical_bs_{}_draws.pkl".format(iters))
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open(fname, "w") as fout:
        cPickle.dump(results, fout)

    print("LIGHTCURVES: Complete ({:.2f}s)                                   " \
        .format(time.time() - tstart))

    return results


def plot_bootstrapping(lc, results):
    """"""
    fig, [r1, r2, r3] = plt.subplots(3, 3, figsize=(6, 6))
    fig.delaxes(r3[2])
    for season, c in zip(["winter", "summer"], ["#56B4E9", "#E69F00"]):
        var = [[res[ii] for res in results[season]] for ii in range(7)]
        params = [[res[ii] for res in var[3]] for ii in range(4)]
        r1[0].hist(var[0], bins=21, color=c, alpha=0.6, normed=True,
            label=season.title())
        r1[1].hist(var[1], bins=21, color=c, alpha=0.6, normed=True)
        r1[2].hist(var[2], bins=21, color=c, alpha=0.6, normed=True)
        r2[0].hist(var[4], bins=21, color=c, alpha=0.6, normed=True)
        r2[1].hist(var[5], bins=21, color=c, alpha=0.6, normed=True)
        r2[2].hist(var[6], bins=21, color=c, alpha=0.6, normed=True)
        r3[0].scatter(params[0], params[1], color=c, s=2, alpha=0.6)
        r3[1].scatter(params[2], params[3], color=c, s=2, alpha=0.6)
    r3[0].set_xlim(4.5, 5.3)
    r3[0].set_ylim(0.2, 0.27)
    r3[1].set_ylim(1400, 1550)
    _ = [ax.tick_params(axis="both", labelsize=6) for ax in fig.axes]
    titles = [["Sample Mean", "Sample Median", "Sample Std."],
              ["Fitted Burr Dist. Mean", "Fitted Burr Dist. Median",
              "Fitted Burr Dist. Std."], ["Burr Params (c vs. k)",
              "Burr Params (scale vs. loc)" , ""]]
    for rr, title in zip([r1, r2, r3], titles):
        _ = [rr[ii].set_title(title[ii], fontsize=8) for ii in range(3)]
    fig.suptitle("Results from {} Iterations of Empirical Bootstrapping" \
        .format(len(results["winter"])), y=.99)
    r1[0].legend(bbox_to_anchor=(3.28, -1.78))
    plt.tight_layout(w_pad=0.05, rect=[0, 0.03, 1., .95])
    plt.show()


if __name__ == "__main__":
    # -- Results file.
    iters = 1000
    folder = os.path.join(lc.path_outpu, "bootstrapping")
    fname = os.path.join(folder, "empirical_bs_{}_draws.pkl".format(iters))
    # -- Calculate/load bootstrap results.
    if os.path.isfile(fname):
        with open(fname, "r") as infile:
            results = cPickle.load(infile)
    else:
        results = empirical_bootstrap(lc, iters)
    # -- Plot results.
    plot_bootstrapping(lc, results)
