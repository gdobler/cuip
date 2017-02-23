import numpy as np


np.random.seed(123)



def img_points(image, skyline):
	n = 5
	x_rand = np.random.choice(skyline.shape[0], n)

	sl = skyline[x_rand]
	city = np.array([1000]*n)
	sky = np.array([150]*n)

	index = [[skyline, city, sky], x_rand]

	np.save(imgname.split('.')[0] + '_index', index)


if __name__ == '__main__':
	skyline = np.load('img1_skyline.npy')
    imgname = 'img1.raw'
    rawimg = np.fromfile(imgname, np.uint8)
    skl = skyline(rawimg, imgname)
    img_point(rawimg, skl)
    


