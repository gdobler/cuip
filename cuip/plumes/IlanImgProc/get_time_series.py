import numpy as np
import os
import sys
import glob

import pylab as pl
import datetime

from PIL import Image
from skyline import skyline
from get3numbers import image_points


file_path = '/projects/projects/project-uo_visible_plumes/workspace/share/plumes/'
path = glob.glob('*.raw')

image_list = []

for filename in glob.glob(file_path + '*.raw'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)

print image_list
