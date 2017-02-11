#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# -- read in the data
for ii in range(10):
    if ii == 0:
        data = pd.read_csv(os.path.join("output", 
                                        "register_{0:04}.csv".format(ii)))
    else:
        data = data.append(pd.read_csv(os.path.join("output", 
                                                    "register_{0:04}.csv" \
                                                        .format(ii))))
nbad = (np.abs(data.drow) > 20).sum()
nnon = (np.abs(data.drow) == 9999).sum()
print("{0}".format(ii))
print("fraction of bad registration {0}".format(nbad / float(len(data))))
print("fraction not registered {0}\n".format(nnon / float(len(data))))
