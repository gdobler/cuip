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
        try:
            im.set_data(img[::fac, ::fac])
        except:
            im.set_data(np.zeros_like(imgs[0][::fac, ::fac]))
        fig.canvas.draw()
        plt.pause(wait)

    return


# -- read in the data
for ii in range(10, 20):
    if ii == 10:
        data = pd.read_csv(os.path.join("output", 
                                        "register_{0:04}.csv".format(ii)), 
                           parse_dates=["timestamp"])
    else:
        data = data.append(pd.read_csv(os.path.join("output", 
                                                    "register_{0:04}.csv" \
                                                        .format(ii)), 
                                       parse_dates=["timestamp"]))
nbad = (np.abs(data.drow) > 200).sum()
nnon = (np.abs(data.drow) == 9999).sum()
print("{0}".format(ii))
print("fraction of bad registration {0}".format(nbad / float(len(data))))
print("fraction not registered {0}\n".format(nnon / float(len(data))))


# -- check a non-registered images
bind = data.drow == -9999
bad  = data[bind]
ex   = bad[-100:]
imgs = [ut.read_raw(os.path.join(i.fpath, i.fname)) for r, i in ex.iterrows()]

np.random.seed(314)
rind = np.random.rand(len(bad)).argsort()[:100]
rx   = bad.iloc[rind].sort_values(by="timestamp")
imgr = [ut.read_raw(os.path.join(i.fpath, i.fname)) for r, i in rx.iterrows()]
