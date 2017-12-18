from __future__ import print_function

import sys
import time
import cPickle
import scipy.stats
import numpy as np

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


if __name__ == "__main__":
    empirical_bootstrap(lc, 1000)
