#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from uo_tools import read_raw
from scipy.ndimage import imread

# -- open the images
path0  = "/uofs10tb_gpfs/roof/1/2016/03/09/23.00.00"
path1  = "/uofs50tb_gpfs/archive/uods1311/2014/01/02/23.00.13"
fname0 = sorted(os.listdir(path0))[0]
fname1 = sorted(os.listdir(path1))[0]
img0   = imread(os.path.join(path0,fname0))
img1   = read_raw(fname1,path1)

# -- get mean and std
avgs0 = img0.mean(0).mean(0)
avgs1 = img1.mean(0).mean(0)
sigs0  = img0.std((0,1))
sigs1  = img1.std((0,1))

# -- get the number of 5 sigma pixels
bright0 = (img0 > (avgs0 + 5*sigs0)).any(-1).sum()
bright1 = (img1 > (avgs1 + 5*sigs1)).any(-1).sum()
