#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from scipy.ndimage.measurements import label, mean
from cuip.cuip.registration.register import get_reference_image

# -- utilities
nrow = 2160
ncol = 4096
buff = 20

# -- get source mask
labs = os.path.join(os.getenv("CUIP_SUPPLEMENTARY"), "window_labels.out")
srcs = np.zeros((nrow, ncol), dtype=bool)
srcs[buff:-buff, buff:-buff] = np.fromfile(labs, int) \
    .reshape(nrow - 2 * buff, ncol - 2* buff) \
    .astype(bool)

# -- label the sources
labs = label(srcs)

# -- get the reference image
ref = get_reference_image()

# -- get means
t0 = time.time()
vals = np.array([mean(ref[..., i], labs[0], range(1, labs[1]+1)) for i in 
                 [0, 1, 2]]).T
print("Brightness extraction time {0}".format(time.time() - t0))
