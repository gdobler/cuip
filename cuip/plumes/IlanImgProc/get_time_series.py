import numpy as np
import os
import sys
import glob

import pylab as pl
import datetime

from PIL import Image


home = '/projects/projects/project-uo_visible_plumes/workspace/share/plumes/'

index = np.load('/home/cusp/ir729/cuip/cuip/plumes/IlanImgProc/index.npy')
skyline = np.load('/home/cusp/ir729/cuip/cuip/plumes/IlanImgProc/img1_skyline.npy')


#file_path = '/projects/projects/project-uo_visible_plumes/workspace/share/plumes/'
path = glob.glob('*.raw')

image_list = []

def img2D(image):
	im = np.fromfile(image, np.uint8)
	im = im.astype(float)
	im *= 255 / im.max()
	im2d = (im.reshape([2160, 4096, 3]).sum(2) / 3.)
	im2d /= im2d.max()
	return im2d

def get_index(list):
	new_list = []
	for item in list:
		new_list.append(im2d[index[0], index[1]])
        return new_list


#return a list of string with names of directories 
def get_dir_name(dir_path):
	dirs = []
	for item in os.listdir(dir_path):
		if os.path.isdir(item) == True:
			dirs.append(item)
	return dirs

#use the string from the list of directories and create a time vector 
def time_vector(string):
	n_photos = 40
	v = datetime.strptime(string, '%H.%M.%S')
	time_v = v + n_photos * datetime.timedelta(seconds = 10)
	return time_v


def get_images(dir_path, dir_list):
	img = {}
	for item in dir_list:
		fldr = os.path.join(dir_path, item)
		for root, dirs, files in os.walk(fldr):
			img[item] = files

	return img

