import numpy as np
import scipy.ndimage as nd
from skyline import skyline

np.random.seed(123)



def img_points(skyline):
	n = 5
	x_rand = np.random.choice(skyline.shape[0], n)

	sl = skyline[x_rand]
	city = np.array([1000]*n)
	sky = np.array([150]*n)
	

	index = [[sl, city, sky], x_rand]
	print index
	return index
	np.save('index', index)


if __name__ == '__main__':
	skyline = np.load('img1_skyline.npy')
	img_points(skyline)
    


