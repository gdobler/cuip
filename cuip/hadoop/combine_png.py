import os
import numpy as np
from scipy.ndimage import imread

inpath = '/uofs10tb_gpfs/roof/1/2016/03/09/'
outpath = '/home/cusp/mohitsharma44/uo/testcombined/'

# File shape
nrows = 2160
ncols = 4096
ndims = 3

# Number of files to combine
combine = 4

# Get all the files
flist = [os.path.join(inpath, f) for f in os.listdir(inpath) if os.path.isfile(f)]

# Array to store the images
comb_imgs = np.zeros([nrows*combine, ncols, ndims], np.uint8)

# Combine images
for ind, imgs in enumerate(flist[:combine]):
    # By default, mode='RGB' means dtype = np.uint8
    comb_imgs[nrows*ind: nrows*(ind+1), :, :] = imread(imgs, mode='RGB')

comb_imgs.tofile(os.path.join(outpath, '2.raw'))
