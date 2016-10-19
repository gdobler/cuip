"""
detect_match.py
Author: Chris Prince [cmp670@nyu.edu]
Date: 25 May 2016
"""

import os
from math import asin
import numpy as np
import pylab as pl
import cv2
from scipy.ndimage.filters import convolve

# Globals
CAMHEIGHT, CAMWIDTH = (2160, 4096)


def loadRAW(f):
        return np.fromfile(f, dtype=np.uint8).reshape(CAMHEIGHT,CAMWIDTH,3)[:,:,::-1]

