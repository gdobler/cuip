import multiprocessing as mpc
import itertools
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pylab as pl
from findImageSize import findsize
from scipy.ndimage.filters import gaussian_filter as gf

#from plm_images import *
from sub_foregroundFed import *

def getsubimseq(lim=LIM, nmax=-1):
    fl = (open(os.path.join(DST_WRITE, 'filelist.txt'), 'rb')).readlines()
    raw = RawImages(fl=fl, lim=lim)    
    lim = LIM-WINDOW if LIM > 0 else len(fl)-WINDOW
    if nmax>0 and lim>nmax:
        lim = nmax + WINDOW

    # -- initialize the the difference image list
    difs = np.zeros([lim] + list(raw.imgs[0].shape))
    #difs = mpc.Array('f', difs.flatten())

    # -- loop through the images
    nps = min(mpc.cpu_count() - 1 or 1, MAXPROCESSES)

    print("before ",difs[6])
    goods = []

    for ii in range(WINDOW, lim):
        try:
            print(ii)
            difs[ii] = np.load(OUTPUTDIR + "/" + 'tmp' + '_%04d.npy'%ii)
            goods.append(ii)
        except IOError:
            pass
        
    difs = difs[WINDOW:]        
    return difs, goods

 

if __name__=='__main__':
    # -- get the file list

    difs, goods = getsubimseq(lim)
    
    '''
    # -- save figures to file
    im = pl.imshow(difs[0][:,:,0],clim=[0,5])
    for ii in goods: #range(len(difs)):
        print(ii)
        im.set_data(difs[ii-WINDOW][:,:,0])
        im.set_clim([0,5])
        pl.draw()
        pl.show()
        pl.savefig(OUTPUTDIR + '/difs_'+str(ii).zfill(3)+'.png',clobber=True)
    '''   
    # -- save panel figures to file
    pl.figure(2,figsize=[15,15])
    pl.subplot(211) 
    raw.readraws()
    imt = pl.imshow(raw.imgs[WINDOW])
    pl.axis('off')
    pl.subplot(212)
    imb = pl.imshow(difs[0][:,:,0],clim=[0,5])
    pl.axis('off')
    pl.subplots_adjust(0.05,0.05,0.95,0.95,0.05,0.05)
#for ii,jj in enumerate(range(28,42)):

    for ii,jj in enumerate(goods): #range(5, lim)):
        print(ii, )
        imt.set_data(raw.imgs[jj])
        pl.draw()
        imb.set_data(difs[ii][:,:,0])
        #    imb.set_data(difs[jj-5][:,:,0])
        pl.clim([0,5])
        pl.draw()
        pl.savefig(OUTPUTDIR + '/difs_imgs_' + str(ii).zfill(3) + 
                   '.png',clobber=True)



