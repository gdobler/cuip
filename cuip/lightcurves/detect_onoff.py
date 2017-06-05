#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
execfile("split_days.py")

# -- utilities
file_index   = 0
width        = 30
delta        = 2
sig_clip_amp = 2.0
sig_peaks    = 10.0

# -- read in lightcurves and convert to grayscale
print("reading lightcurves for nights index {0}".format(file_index))
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
lcs_gd = np.ma.zeros(lcs_sm.shape, dtype=lcs_sm.dtype)
lcs_gd[delta // 2: -delta // 2] = lcs_sm[delta:] - lcs_sm[:-delta]

# -- set the gaussian difference mask
lcs_gd.mask = np.zeros_like(msk)
lcs_gd.mask[delta // 2: -delta // 2] = ~(msk[delta:] * msk[:-delta])

# -- get the indices of the date separators and create the individual dates
dind_lo = list(split_days(file_index))
dind_hi = dind_lo[1:] + [lcs_gd.shape[0]]
nights  = [lcs_gd[i:j] for i, j in zip(dind_lo, dind_hi)]

# -- sigma clip and reset the means, standard deviations, and masks
avgs = []
sigs = []
for ii in range(len(nights)):
    print("working on night {0}".format(ii))
    tmsk = nights[ii].mask.copy()
    for _ in range(10):
        avg             = nights[ii].mean(0)
        sig             = nights[ii].std(0)
        nights[ii].mask = np.abs(nights[ii] - avg) > sig_clip_amp * sig
    avgs.append(nights[ii].mean(0).data)
    sigs.append(nights[ii].std(0).data)
    nights[ii].mask = tmsk

# -- tag the potential ons and offs


def canny1d(lcs, indices=None, width=30, delta=2, see=False, sig_clip_iter=10, 
            sig_clip_amp=2.0, sig_peaks=10.0, xcheck=True, sig_xcheck=2.0):



        # -- find peaks in RGB
        ind_on_rgb, ind_off_rgb = [], []

        tags_on  = (dlcg-avg > sig_peaks*sig) & \
            (dlcg>np.roll(dlcg,1,0)) & \
            (dlcg>np.roll(dlcg,-1,0)) & \
            ~dlcg.mask

        tags_off = (dlcg-avg < -sig_peaks*sig) & \
            (dlcg<np.roll(dlcg,1,0)) & \
            (dlcg<np.roll(dlcg,-1,0)) & \
            ~dlcg.mask

        for band in [0,1,2]:
            ind_on_rgb.append([i for i in ints[tags_on[:,band]]])
            ind_off_rgb.append([ i for i in ints[tags_off[:,band]]])


        # -- collapse RGB indices
        for iind in ind_on_rgb[0]:
            for jind in ind_on_rgb[1]:
                if abs(iind-jind)<=2:
                    ind_on_rgb[1].remove(jind)
            for jind in ind_on_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_on_rgb[2].remove(jind)

        for iind in ind_on_rgb[1]:
            for jind in ind_on_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_on_rgb[2].remove(jind)

        ind_on_list = ind_on_rgb[0] + ind_on_rgb[1] + ind_on_rgb[2]

        for iind in ind_off_rgb[0]:
            for jind in ind_off_rgb[1]:
                if abs(iind-jind)<=2:
                    ind_off_rgb[1].remove(jind)
            for jind in ind_off_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_off_rgb[2].remove(jind)

        for iind in ind_off_rgb[1]:
            for jind in ind_off_rgb[2]:
                if abs(iind-jind)<=2:
                    ind_off_rgb[2].remove(jind)

        ind_off_list = ind_off_rgb[0] + ind_off_rgb[1] + ind_off_rgb[2]


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
