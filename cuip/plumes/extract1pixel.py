import glob
import pylab as pl
import numpy as np
import os
from findImageSize import findsize

from utils import RawImages


files = (glob.glob(os.getenv("PLUMESDIR") + "/*raw"))
imsize = findsize(files[0], outputdir=os.getenv("PLUMESDIR"),
                  imsizefile = "oct08_2013-10-25_imgsize.txt")


pixels = np.array([(50,50), (100,20)])
pixvalues = np.zeros((len(files), pixels.shape[0], 3)) * np.nan



for i, f in enumerate(files):
    pixvalues[i] = RawImages(fl=[f], lim=1, imsize=imsize, pixels=pixels).pixvals


print (pixvalues)

