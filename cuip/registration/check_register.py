#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import uo_tools as ut
import matplotlib.pyplot as plt

def view_images(imgs, fac=4, wait=1e-3):
    plt.close("all")

    xs = 8.0
    ys = xs * float(imgs[0].shape[0]) / float(imgs[0].shape[1])
    fig, ax = plt.subplots(figsize=(xs, ys))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    im = ax.imshow(imgs[0][::fac, ::fac])
    fig.canvas.draw()
    plt.ion()
    plt.show()

    for img in imgs:
        im.set_data(img[::fac, ::fac])
        fig.canvas.draw()
        plt.pause(wait)

    return


def view_random(df, nframe=100, fac=4, wait=1e-3):
    np.random.seed(314)
    rind = np.random.rand(len(df)).argsort()[:nframe]
    rx   = df.iloc[rind].sort_values(by="timestamp")

    img0 = ut.read_raw(rx.iloc[0].fpath, rx.iloc[0].fname)

    plt.close("all")

    xs = 8.0
    ys = xs * float(img0.shape[0]) / float(img0.shape[1])
    fig, ax = plt.subplots(figsize=(xs, ys))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    im = ax.imshow(img0[::fac, ::fac])
    fig.canvas.draw()
    plt.ion()
    plt.show()

    for ii in range(nframe):
        try:
            im.set_data(ut.read_raw(rx.iloc[ii].fpath, 
                                    rx.iloc[ii].fname)[::fac, ::fac])
        except:
            continue
        fig.canvas.draw()
        plt.pause(wait)

    return


def view_flist(flist, fac=4, wait=1e-3):
    plt.close("all")

    img = ut.read_raw(flist[0])

    xs = 8.0
    ys = xs * float(img.shape[0]) / float(img.shape[1])
    fig, ax = plt.subplots(figsize=(xs, ys))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    im = ax.imshow(img[::fac, ::fac])
    fig.canvas.draw()
    plt.ion()
    plt.show()

    for tfile in flist:
        im.set_data(ut.read_raw(tfile)[::fac, ::fac])
        fig.canvas.draw()
        plt.pause(wait)

    return


# -- read in the data
#for ii in range(10):
#    if ii == 0:
for ii in range(10,20):
    if ii == 10:
        data = pd.read_csv(os.path.join("output", 
                                        "register_{0:04}.csv".format(ii)), 
                           parse_dates=["timestamp"])
    else:
        data = data.append(pd.read_csv(os.path.join("output", 
                                                    "register_{0:04}.csv" \
                                                        .format(ii)), 
                                       parse_dates=["timestamp"]))
data.reset_index(inplace=True)
nbad = (np.abs(data.drow) > 20).sum()
nnon = (np.abs(data.drow) == 9999).sum()
print("{0}".format(ii))
print("fraction of bad registration {0}".format(nbad / float(len(data))))
print("fraction not registered {0}\n".format(nnon / float(len(data))))


# -- check a non-registered images
bind = data.drow == -9999
bad  = data[bind]
ex   = bad[-100:]
# imgs = [ut.read_raw(os.path.join(i.fpath, i.fname)) for r, i in ex.iterrows()]

np.random.seed(314)
rind = np.random.rand(len(bad)).argsort()[:100]
rx   = bad.iloc[rind].sort_values(by="timestamp")
# imgr = [ut.read_raw(os.path.join(i.fpath, i.fname)) for r, i in rx.iterrows()]


# -- select some poorly registered frames
sub = data[24000:26000]
sub = sub[(sub.drow > -9999) & (sub.drow < -50)]

