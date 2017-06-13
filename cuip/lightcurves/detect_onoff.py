#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
from scipy.ndimage import correlate1d
from split_days import *

# -- utilities
file_index   = 0
width        = 30
delta        = 2
sig_clip_amp = 2.0
sig_peaks    = 10.0
sig_xcheck   = 2.0

# -- read in lightcurves and convert to grayscale
print("reading lightcurves for nights index {0}...".format(file_index))
lcs = np.load(os.path.join("output", "light_curves_{0:04}.npy" \
                               .format(file_index))).mean(-1)

# -- generate a mask
print("generating mask...")
msk = gf((lcs > -9999).astype(float), (width, 0)) > 0.9999

# -- convert to smooth the lightcurves (taking into account the mask)
print("smoothing lightcurves...")
msk_sm = gf(msk.astype(float), (width, 0))
lcs_sm = gf(lcs * msk, (width, 0)) / (msk_sm + (msk_sm == 0))

# -- compute the gaussian difference (using a masked array now)
print("computing gaussian differences...")
lcs_gd = np.ma.zeros(lcs_sm.shape, dtype=lcs_sm.dtype)
lcs_gd[delta // 2: -delta // 2] = lcs_sm[delta:] - lcs_sm[:-delta]

# -- set the gaussian difference mask
print("resetting the mask...")
lcs_gd.mask = np.zeros_like(msk)
lcs_gd.mask[delta // 2: -delta // 2] = ~(msk[delta:] * msk[:-delta])

# -- get the indices of the date separators and create the individual dates
print("splitting dates...")
dind_lo = list(split_days(file_index))
dind_hi = dind_lo[1:] + [lcs_gd.shape[0]]
nights  = [lcs_gd[i:j] for i, j in zip(dind_lo, dind_hi)]
nnights = len(nights)

# -- sigma clip and reset the means, standard deviations, and masks
avgs = []
sigs = []
for ii in range(nnights):
    print("sigma clipping night {0} of {1}...".format(ii + 1, nnights))
    tmsk = nights[ii].mask.copy()
    for _ in range(10):
        avg             = nights[ii].mean(0)
        sig             = nights[ii].std(0)
        nights[ii].mask = np.abs(nights[ii] - avg) > sig_clip_amp * sig
    avgs.append(nights[ii].mean(0).data)
    sigs.append(nights[ii].std(0).data)
    nights[ii].mask = tmsk

# -- tag the potential ons and offs
tags_on  = [np.zeros(i.shape, dtype=bool) for i in nights]
tags_off = [np.zeros(i.shape, dtype=bool) for i in nights]

for ii in range(nnights):
    print("finding extrema for night {0} of {1}".format(ii + 1, nnights))
    tags_on[ii][1:-1]  = (nights[ii] - avgs[ii] > 
                          sig_peaks * sigs[ii])[1:-1] & \
                          (nights[ii][1:-1] > nights[ii][2:]) & \
                          (nights[ii][1:-1] > nights[ii][:-2]) & \
                          ~nights[ii].mask[1:-1]
    tags_off[ii][1:-1] = (nights[ii] - avgs[ii] < 
                          -sig_peaks * sigs[ii])[1:-1] & \
                          (nights[ii][1:-1] < nights[ii][2:]) & \
                          (nights[ii][1:-1] < nights[ii][:-2]) & \
                          ~nights[ii].mask[1:-1]

# -- cross check left/right means for robustness to noise
print("setting up filters...")
mean_diff  = ((np.arange(2 * width) >= width) * 2 - 1) / float(width)
mean_left  = 1.0 * (np.arange(2 * width) < width) / float(width)
mean_right = 1.0 * (np.arange(2 * width) >= width) / float(width)
print("calculating mean difference across transition...")
lcs_md     = np.abs(correlate1d(lcs, mean_diff, axis=0))
print("calculating max standard deviation across transition...")
lcs_sq     = lcs**2
lcs_std    = np.sqrt(np.maximum(correlate1d(lcs_sq, mean_left, axis=0) - 
                                correlate1d(lcs, mean_left, axis=0)**2, 
                                correlate1d(lcs_sq, mean_right, axis=0) - 
                                correlate1d(lcs, mean_right, axis=0)**2))

# -- identify all potentially robust transitions and prune list
#    NB: careful about handling partial initial days.
print("pruning ons and offs for statistical significance...")
pad       = np.zeros((dind_lo[0], lcs.shape[1]), dtype=bool)
good_arr  = lcs_md > sig_xcheck * lcs_std
good_ons  = np.vstack([pad] + tags_on) & good_arr
good_offs = np.vstack([pad] + tags_off) & good_arr


def canny1d(lcs, indices=None, width=30, delta=2, see=False, sig_clip_iter=10, 
            sig_clip_amp=2.0, sig_peaks=10.0, xcheck=True, sig_xcheck=2.0):



        # -- cross check left/right means for robustness to noise
        if xcheck:
            rtwd = np.sqrt(width)

            for on in [_ for _ in ind_on_list]:
                mn_l  = lcs.lcs[index,on-width:on].mean(1).mean()
                err_l = lcs.lcs[index,on-width:on].mean(1).std()
                mn_r  = lcs.lcs[index,on:on+width].mean(1).mean()
                err_r = lcs.lcs[index,on:on+width].mean(1).std()

                if abs(mn_r-mn_l)<(sig_xcheck*max(err_l,err_r)):
                    ind_on_list.remove(on)

            for off in [_ for _ in ind_off_list]:
                mn_l  = lcs.lcs[index,off-width:off].mean(1).mean()
                err_l = lcs.lcs[index,off-width:off].mean(1).std()
                mn_r  = lcs.lcs[index,off:off+width].mean(1).mean()
                err_r = lcs.lcs[index,off:off+width].mean(1).std()

                if abs(mn_r-mn_l)<(sig_xcheck*max(err_l,err_r)):
                    ind_off_list.remove(off)


        # -- add to on/off list
        tind_onoff = np.array([i for i in ind_on_list+[-j for j in 
                                                        ind_off_list]])

        ind_onoff.append(tind_onoff[np.argsort(np.abs(tind_onoff))])

#        if see:
#            plt.subplot(2,2,2)
#            plt.plot(np.arange(dlcg.shape[0])[on_ind[:,0]],
#                     dlcg[on_ind[:,0],0], 'go')


    return ind_onoff
