import numpy as np
import os
import sys
import glob

import pylab as pl
import datetime

from PIL import Image

#change and revise
index = np.load('~/cuip/cuip/plumes/IlanImgProc/index.npy', mmap_mode = 'r')
skyline = np.load('../img1_skyline.npy')


file_path = '/projects/projects/project-uo_visible_plumes/workspace/share/plumes/'
path = glob.glob('*.raw')

image_list = []

for file in path: #assuming gif
    im = np.fromfile(file, np.uint8)
    im = im.astype(float)
    im *= 255 / im.max()
    im2d = im.reshape([2160, 4096, 3]).sum(2) / 3.)
    im2d /= im2d.max()
    image_list.append(im2d)

img_index = []
for item in image_list:
    img_index.append(im2d[index[0], index[1]])

base = datetime.datetime(100,1,1,10,51,4)

date_list = base + np.arange(len(img_index)) * datetime.timedelta(seconds = 10) 
