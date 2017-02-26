import multiprocessing as mpc
import itertools
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pylab as pl
from findImageSize import findsize
from scipy.ndimage.filters import gaussian_filter as gf
from utils import RawImages
from configs import *
#from plm_images import *


    
def subforg(difs, allframes, i, j, fname):
#image loop
    #lf must be odd
    fulllen = np.prod(allframes[0].shape)
    frames = allframes[i-j:i+j+1]
    lf = int(len(frames) / 2)
    
    #print('working on {0}'.format(i))
    dif = abs(- 1.0 * np.concatenate([frames[:lf], 
                                   frames[(lf+1):]]) + 1.0 * frames[lf]).min(0)
    ##for jj, fr in enumerate(frames):
    ##    dif[jj] = 1.0 * raw.imgs[ii] - 1.0 * raw.imgs[fr]
    #print(i,'here', frames.shape, dif.shape)#[0,0,0,:5])
    #tmp = abs(dif).min(0)
    #difs[i*fulllen: i*fulllen+len(tmp.flatten())] = tmp.flatten()
    np.save(OUTPUTDIR + "/" + fname + '_%04d.npy'%i, dif)

if __name__=='__main__':

    # -- get the file list
    #fl = pkl.load(open(os.path.join(os.environ['DST_WRITE'],'filelist.pkl'),'rb'))
    fl = (open(os.path.join(DST_WRITE, 'filelist.txt'), 'rb')).readlines()

    # -- get the raw images
    raw = RawImages(fl=fl, lim=LIM)
    # set the image numbers
    lim = LIM-WINDOW if LIM > 0 else len(fl)-WINDOW

    # -- initialize the the difference image list
    difs = np.zeros([lim] + list(raw.imgs[0].shape))
    #difs = mpc.Array('f', difs.flatten())

    # -- loop through the images
    nps = min(mpc.cpu_count() - 1 or 1, MAXPROCESSES)

    #pool = mpc.Pool(processes=nps)
#print([raw.imgs[i-5:i+6].flatten().shape for i in range(5, lim)])
#sys.exit()
#tmp = pool.map(subforg, itertools.izip([raw.imgs[i-5:i+6] for i in range(5, lim)], 
#              itertools.repeat(5)))
#pool.close
    print("before ",difs[6])
    processes = [mpc.Process(target=subforg, args=(difs, raw.imgs, i, WINDOW, 'tmp')) for i in range(WINDOW, lim)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    
    #difs = np.array(difs[:]).reshape([lim] + list(raw.imgs[0].shape))
    #print("after", difs[6])

#Parallel(n_jobs=nps)(
#        delayed(sum(raw.imgs[i-5:i+6]) for i in range(5, lim)))
#tmp = np.zeros(100)
#difs = tmp

#calc, itertools.izip(range(NM0, nm), itertools.repeat(second_args)))  # for i in range(nm): result[i] = f(i, second_args)

#for ii in range(5, lim):
#    difs[ii] = subforg(ii, raw, difs[ii])
#    difs = difs[5:]
'''
# -- save figures to file
im = pl.imshow(difs[0][:,:,0],clim=[0,5])
for ii in range(len(difs)):
im.set_data(difs[ii][:,:,0])
im.set_clim([0,5])
pl.draw()
pl.show()
pl.savefig(OUTPUTDIR + '/difs_'+str(ii).zfill(3)+'.png',clobber=True)


# -- save panel figures to file
pl.figure(2,figsize=[15,15])
pl.subplot(211)
imt = pl.imshow(raw.imgs[5])
pl.axis('off')
pl.subplot(212)
imb = pl.imshow(difs[0][:,:,0],clim=[0,5])
pl.axis('off')
pl.subplots_adjust(0.05,0.05,0.95,0.95,0.05,0.05)
#for ii,jj in enumerate(range(28,42)):
for ii,jj in enumerate(range(WINDOW, lim)):
imt.set_data(raw.imgs[jj])
pl.draw()
imb.set_data(difs[ii][:,:,0])
#    imb.set_data(difs[jj-WINDOW][:,:,0])
pl.clim([0,5])
pl.draw()
pl.savefig(OUTPUTDIR + '/difs_imgs_' + str(ii).zfill(3) + 
'.png',clobber=True)

'''

