import numpy as np
import scipy.ndimage as nd
from skyline import skyline

np.random.seed(100)



def img_points(skyline):
	n = 5
	x_rand = np.random.choice(skyline.shape[0], n)

	sl = skyline[x_rand]
	city = [1000]*n
	sky = [150]*n
	

	sl_idx  = [sl, x_rand]
	city_idx = [city, x_rand]
	sky_idx = [sky, x_rand]
	
	np.save('skyline_idx', sl_idx)
	np.save('city_idx', city_idx)
	np.save('sky_idx', sky_idx)

	return sl_idx, city_idx, sky_idx

if __name__ == '__main__':
	skyline = np.load('img1_skyline.npy')
	img_points(skyline)
    


