from __future__ import print_function

import os
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

OUTP = os.environ["OUTPATH"]

# -- Select relevant nights without masked values.
wkdays = lc.lc_bigoffs[(lc.lc_bigoffs.index.dayofweek < 5) &
    (lc.lc_bigoffs.index.isin(lc.meta[lc.meta.nmask == 0].index)) &
    (~lc.lc_bigoffs.index.isin([pd.datetime(2013, 10, 31),
                                pd.datetime(2013, 12, 30)]))].drop_duplicates()
winter = wkdays[(wkdays.index.month > 9)]
summer = wkdays[wkdays.index.month < 9]
# -- Select residential sources and take the daily median.
res_labs = filter(lambda x: lc.coords_cls[x] == 1, lc.coords_cls.keys())
winter_vals = winter[np.array(res_labs) - 1].median(axis=1).values
summer_vals = summer[np.array(res_labs) - 1].median(axis=1).values
# -- Plot boxplot.
fig, ax = plt.subplots(figsize=(8, 2))
ax.boxplot([winter_vals, summer_vals], vert=False, positions=[0, 0.2],
    labels=["Winter", "Summer"])
ax.set_title("Median Weekday Res. Bigoff Timestep for Summer and Winter")
ax.set_xlabel("Timesteps")
ax.set_ylim(-0.1, 0.3)
plt.tight_layout()
plt.show(block=True)


# -- Plot winter v. summer histograms.
winter_vals = [val for val in winter.values.ravel() if ~np.isnan(val)]
winter_vals = np.array(winter_vals)
summer_vals = [val for val in summer.values.ravel() if ~np.isnan(val)]
summer_vals = np.array(summer_vals)
# -- Bins and bin middles for histograms.
x_bins = np.arange(0, 3000, 100)
x_middles = x_bins[1:] - 50
# -- Fitted distribution.
dist_names = ["burr"] # ['beta', 'burr', 'gausshyper', 'kstwobign',  'mielke', 'nakagami']
results = []
# -- Plot
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, bigoffs, c in [[ax1, winter_vals, "#56B4E9"], [ax2, summer_vals, "#E69F00"]]:
    # -- Plot summer/winter hist, mean, and median.
    ax.hist(bigoffs, x_bins, normed=True, color="gray")
    ax.axvline(bigoffs.mean(), c="k", ls="dashed")
    ax.axvline(np.median(bigoffs), c="k", ls=":")
    # -- For each distribution, fit for both summer and winter.
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(bigoffs)
        pdf = dist.pdf(x_middles, *param)
        results.append([dist_name, pdf])
        # -- Plot fitted models on histogram and on ax3.
        for ax_ in [ax, ax3]:
            ax_.axvline(dist.mean(*param), c=c, ls="dashed")
            ax_.axvline(dist.median(*param), c=c, ls=":")
            ax_.plot(x_middles, pdf,  c=c)
# -- Compare histograms of summer v. winter bigoffs.
ks_res = scipy.stats.ks_2samp(winter_vals, summer_vals)
mw_res = scipy.stats.mannwhitneyu(winter_vals, summer_vals)
resampled_wbig = np.random.choice(winter_vals, len(summer_vals), False)
en_res = scipy.stats.entropy(summer_vals, resampled_wbig)
# -- Compare fitted models of summer v. winter bigoffs.
ks_mod = scipy.stats.ks_2samp(results[0][1], results[1][1])
mw_mod = scipy.stats.mannwhitneyu(results[0][1], results[1][1])
en_mod = scipy.stats.entropy(results[0][1], results[1][1])
res = [[ks_res, mw_res, en_res], [ks_mod, mw_mod, en_mod]]
# -- Text
text = ["Entropy: {:.4f}", "KS-test (p-value): {:.4f}",
        "Mann-Whitney (p-value): {:.4f}"]
for ax, res_ in zip([ax1, ax3], res):
    ax.text(20, 0.00007, text[0].format(res_[2]), size=8)
    ax.text(20, 0.00004, text[1].format(res_[0].pvalue), size=8)
    ax.text(20, 0.00001, text[2].format(res_[1].pvalue), size=8)
ax1.text(20, 0.0001, "Histogram comparison:", size=8)
ax3.text(20, 0.0001, "Model comparison:", size=8)
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Timesteps")
    ax.set_xlim(0, 2700)
    ax.set_ylim(0, 0.00085)
solid = plt.axvline(-1, c="gray")
dash = plt.axvline(-1, ls="dashed", c="gray")
dot = plt.axvline(-1, ls=":", c="gray")
ax1.set_ylabel("Counts (Normed)")
ax1.set_title("Winter Bigoffs")
ax2.set_title("Summer Bigoffs")
ax3.set_title("Fitted Burr Distributions")
ax3.legend([dash, dot, solid], ["Mean", "Median", "Fitted Burr Dist."])
plt.tight_layout()
plt.show()
