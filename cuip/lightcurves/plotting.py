#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def scroll_onoff_detections(nights, ons, offs):
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

    def scroll_plot(event):
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
            nght[0] -= 1
    
        # -- handle negative values
        nght[0] = nght[0] % len(nights)
        isrc[0] = isrc[0] % nights[0].shape[1]
    
        # -- update the data values
        src_ons  = indices[nght[0]][ons[nght[0]][:, isrc[0]]]
        src_offs = indices[nght[0]][offs[nght[0]][:, isrc[0]]]
        ax.set_xlim(0, indices[nght[0]].size)
    
        # -- update plot
        lin.set_data(indices[nght[0]], nights[nght[0]][:, isrc[0]])
        pts_on.set_data(src_ons, nights[nght[0]][src_ons, isrc[0]])
        pts_off.set_data(src_offs, nights[nght[0]][src_offs, isrc[0]])
        ax.set_title("src {0:5}, night {1:3}".format(isrc[0], nght[0]))
        fig.canvas.draw()
    
        return
    
    # -- utilities
    indices  = [np.arange(i.shape[0]) for i in nights]
    nght     = [0]
    isrc     = [0]
    # src_ons  = indices[ons[nght[0]][:, isrc[0]]]
    # src_offs = indices[offs[nght[0]][:, isrc[0]]]
    src_ons  = indices[nght[0]][ons[nght[0]][:, isrc[0]]]
    src_offs = indices[nght[0]][offs[nght[0]][:, isrc[0]]]
    
    # -- initialize the plot
    fig, ax  = plt.subplots()
    lin,     = ax.plot(indices[nght[0]], nights[nght[0]][:, isrc[0]])
    pts_on,  = ax.plot(src_ons, nights[nght[0]][src_ons, isrc[0]], 'go')
    pts_off, = ax.plot(src_offs, nights[nght[0]][src_offs, isrc[0]], 'ro')
    ax.set_ylim(0, 255)
    ax.set_xlim(0, indices[nght[0]].size)
    ax.set_title("src {0:5}, night {1:3}".format(isrc[0], nght[0]))
    
    # -- bind keys to update plot
    fig.canvas.draw()
    fig.canvas.mpl_connect("key_press_event", scroll_plot)
    
    plt.show()
