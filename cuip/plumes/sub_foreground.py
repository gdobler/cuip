import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.ndimage.filters import gaussian_filter as gf
from plm_images import *

# -- get the file list
fl = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'filelist.pkl'),'rb'))


# -- get the raw images
raw = RawImages(fl=fl)


# -- initialize the the difference image list
difs = []
dif  = np.zeros([10]+list(raw.imgs[0].shape))


# -- loop through the images
for ii in range(5,54):
    frames = range(ii-5,ii) + range(ii+1,ii+6)
    print('working on {0}'.format(ii),frames)
    for jj, fr in enumerate(frames):
        dif[jj] = 1.0*raw.imgs[ii]-1.0*raw.imgs[fr]
    difs.append(abs(dif).min(0))


# -- save figures to file
im = imshow(difs[0][:,:,0],clim=[0,5])
for ii in range(len(difs)):
    im.set_data(difs[ii][:,:,0])
    clim([0,5])
    draw()
    savefig('../output/difs_'+str(ii).zfill(3)+'.png',clobber=True)


# -- save panel figures to file
plt.figure(2,figsize=[15,15])
plt.subplot(211)
imt = plt.imshow(raw.imgs[5])
plt.axis('off')
plt.subplot(212)
imb = plt.imshow(difs[0][:,:,0],clim=[0,5])
plt.axis('off')
plt.subplots_adjust(0.05,0.05,0.95,0.95,0.05,0.05)
#for ii,jj in enumerate(range(28,42)):
for ii,jj in enumerate(range(5,54)):
    imt.set_data(raw.imgs[jj])
    plt.draw()
    imb.set_data(difs[ii][:,:,0])
#    imb.set_data(difs[jj-5][:,:,0])
    plt.clim([0,5])
    plt.draw()
    plt.savefig('../output/difs_imgs_'+str(ii).zfill(3)+'.png',clobber=True)

