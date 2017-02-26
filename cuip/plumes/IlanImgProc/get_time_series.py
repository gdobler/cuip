import numpy as np
import os
import os.path
import glob
im
import pylab as pl
import datetim as dt


from PIL import Image


home = '/projects/projects/project-uo_visible_plumes/workspace/share/plumes/'
root = '/home/cusp/ir729/cuip/cuip/plumes/IlanImgProc/'

sl_idx = np.load(root + 'skyline_idx.npy')
city_idx  = np.load(root + 'city_idx.npy')
sky_idx = np.load(root + 'sky_idx.npy')
skyline = np.load(root + 'img1_skyline.npy')

dir_gen = os.walk(home)

path = glob.glob('*.raw')


def img2D(file_path):
	im = np.fromfile(file_path, np.uint8)
	im = im.astype(float)
	im *= 255 / im.max()
	im2d = (im.reshape([2160, 4096, 3]).sum(2) / 3.)
	im2d /= im2d.max()
	return im2d


#Still need tweaking and revision
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
#generate a dictioary with the name of the subdirectories and the corresponding timestamp to each image
def time_tree(generator):
	n_photos = 40
	time_tree = {}
	for root_path, dirs, files in generator:
		time = os.path.basename(root_path)
		base = datetime.datetime.strptime(time, '%H.%M.%S')
		time_tree[time] = [base + datetime.timedelta(seconds = i*10) for i in range(0,n_photos)]
		
	return time_tree


def get_images(generator):
	path_tree = {}
	for root_path, dirs, files in generator:
		path_tree[os.path.basename(root_path)] = [os.path.join(root_path, file_path) for file_path in files]
	return path_tree

def read_img(dictionary):
	img_tree = {}
	for key in dictionary:
		img_list = []
		for item in dictionary[key]:
			im = im2D(item)
			img_list.append(im)
		img_tree[key] = img_list
		
	return img_tree


if __name__ == '__main__':
	home = '/projects/projects/project-uo_visible_plumes/workspace/share/plumes/'
	root = '/home/cusp/ir729/cuip/cuip/plumes/IlanImgProc/'

	sl_idx = np.load(root + 'skyline_idx.npy')
	city_idx  = np.load(root + 'city_idx.npy')
	sky_idx = np.load(root + 'sky_idx.npy')
	skyline = np.load(root + 'img1_skyline.npy')
	dir_gen = os.walk(home)

	
	
	t_dict =  time_tree(dir_gen)
	img_dict = get_images(dir_gen)
	img_vector = read_img(img_dict)

#At this point I should still get the indexes to each image and then merge both dictionaries 
#in such a way that I hava dataframe that I can use to create the timeseries
