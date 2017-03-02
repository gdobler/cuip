#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from assign_bbls import *

def get_subset(bbl=1002440019):
    """
    Get the indices of the sources for a given BBL.
    """

    # -- get the bbls
    bbls = assign_bbls()

    return np.arange(bbls.size)[bbls == bbl]
