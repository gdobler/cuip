#Import packages and libraries

import os
import sys
import numpy as np
import pylab as pl
import scipy.ndimage as nd
from skimage import feature
from skimage import filters as skfl
import matplotlib.pyplot as plt



#Define showme() function 

def showme(image, ax=None, cmap=None):
    if ax is None:
        ax = pl.figure(figsize=(11,11)).add_subplot(111)

    if cmap is None:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap=cmap)
    ax.axis('off')

def skyline(image, imgname, plotme=False):

	#Convert the image to float
	img = rawimg.astype(float)

	#Increase contrast
	img *= 255 / img.max() 

	#Reshape from 3D to 2D, normalize
	img2d = (img.reshape([2160, 4096, 3]).sum(2) / 3.)
	img2d /= img2d.max()

	#In order to extract the skyline we smooth the image
	#by using a Gaussian filter. 
	smoothImg = nd.filters.gaussian_filter(img2d, [8, 8])

	#Now we brighten and highlight the upper half of the image
	#by using an increasing array.
	smoothImgEn = smoothImg[:,:] * (np.atleast_2d(np.linspace(1, 0, smoothImg.shape[0])).T)**2

	#We now apply a Sobel filter to find the edges 
	#of the image so we can identify the skyline pixels.
	edge = skfl.sobel(smoothImgEn)

	#In order to start filtering lines from the clouds and 
	#the lower part of the buildings we filter the image
	#by making zero those pixels that are under a threshold value
	#this value was determined by analyzing the historgram the image

	edge[edge < 0.0025] = 0.0
	#showme(edge, cmap = 'gray')

	#Now we can select a first skyline by breaking the image column-wise
	#and taking the gradient of each column. Then the gradient is sorted in such way that the 
	#we select the value that is closest to the top of the image. We select the top 20
	#values and select the minimum.

	rows = edge.shape[0]
	cols = edge.shape[1]
	
	grad_max = []
	for i in range(cols):
		grad = np.gradient(edge[:,i])
		grad_max.append(np.argpartition(grad, -20)[-20:].min())

        if plotme:
            showme(edge, cmap = 'gray')
            plt.plot(np.arange(cols), grad_max, 'g-', ms = 1.5)
            plt.show()
	np.save(imgname.split('.')[0] + '_skyline', grad_max)
	return grad_max


#Read image

#path = 'cuip/cuip/plumes/IlanImgProc/'

if __name__ == '__main__':
    imgname = 'img1.raw'
    rawimg = np.fromfile(imgname, np.uint8)
    skl = np.load('img1_skyline.npy')
    print (skl)



