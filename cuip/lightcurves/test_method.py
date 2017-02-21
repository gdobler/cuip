#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage.measurements import label, mean
from cuip.cuip.registration.register import get_reference_image

# -- utilities
nrow = 2160
ncol = 4096
buff = 20

# -- get source mask
srcs = np.zeros((nrow, ncol), dtype=bool)
srcs[buff:-buff, buff:-buff] = np.fromfile("window_labels.out", int) \
    .reshape(nrow - 2 * buff, ncol - 2* buff) \
    .astype(bool)

# -- label the sources
labs = label(srcs)

# -- get the reference image
ref = get_reference_image()

# -- get means
vals = np.array([mean(ref[..., i], labs[0], range(1, labs[1]+1)) for i in 
                 [0, 1, 2]]).T
