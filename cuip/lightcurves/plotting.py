#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def scroll_onoff_detections():
    """
    Scroll through on/off detections for lightcurves.

    Parameters
    ----------
    nights : list
        List of arrays of lightcurves.
    ons : list
        List of arrays of on detections.
    offs : list
        List of arrays of off detections.
    """

    def scroll_plot():
        """
        Scroll through lightcurve examples.
        """

        # -- set change type
        key = event.key

        if key == "right":
            isrc[0] += 1
        elif key == "left":
            isrc[0] -= 1
        elif key == "up":
            nght[0] += 1
        elif key == "down":
            nght[1] -= 1

        # -- update the data values
        src_ons  = indices(ons[nght][:, isrc])
        src_offs = indices(offs[nght][:, isrc])
        ax.set_xlim(0, indices.size)

        # -- update plot
        lin.set_data(indices[nght], nights[nght][:, isrc])
        pts_on,  = ax.plot(src_ons, nights[nght][src_ons, isrc])
        pts_off, = ax.plot(src_ons, nights[nght][src_offs, isrc])
        fig.canvas.draw()

        return

    # -- utilities
    indices  = [np.arange(i.shape[0]) for i in nights]
    nght     = [0]
    isrc     = [0]
    src_ons  = indices(ons[nght][:, isrc])
    src_offs = indices(offs[nght][:, isrc])

    # -- initialize the plot
    fig, ax  = plt.subplots()
    lin,     = ax.plot(indices[nght], nights[nght][:, isrc])
    pts_on,  = ax.plot(src_ons, nights[nght][src_ons, isrc])
    pts_off, = ax.plot(src_ons, nights[nght][src_offs, isrc])
    ax.set_ylim(0, 255)
    ax.set_xlim(0, indices.size)

    # -- bind keys to update plot
    fig.canvas.draw()
    fig.canvas.mpl_connect("key_press_event", scroll_plot())

    plt.show()
