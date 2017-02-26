import numpy as np
from configs import *

class RawImages:
    def __init__(self, fl=None,  lim=-1, imsize = None, pixels=None):
        
        self.imsize = imsize
        self.pixels = pixels
        fl = [f.strip() for f in fl if f.strip().endswith('.raw')][:lim]
        print (fl, lim)
        fl0 = os.path.join(DST_WRITE,fl[0])
        
        if self.imsize is None:
            self.imsize = findsize(fl0, 
                      filepattern = '-'.join(fl0.split('/')[-1].\
                                             split('.')[0].split('-')[:-2]),
                               outputdir = OUTPUTDIR)

        self.imgs = np.zeros([len(fl), self.imsize['nrows'], 
                              self.imsize['ncols'],
                              self.imsize['nbands']])    
        for i,f in enumerate(fl):    
            self.imgs[i] = self.readraw(os.path.join(DST_WRITE,f))
        
        if not self.pixels is None:
            self.readvals()
            
    def readraw(self, imfile):
        rgb = np.fromfile(imfile, dtype=np.uint8).clip(0, 255).\
                                reshape(self.imsize['nrows'],
                                        self.imsize['ncols'],
                                        self.imsize['nbands']).astype(float)
        return rgb

    def readvals(self):
        self.pixvals = np.zeros([self.imgs.shape[0],
                                 self.pixels.shape[0],
                                 self.imsize['nbands']]) * np.nan
        for i,rgb in enumerate(self.imgs):
            for j, pixels in enumerate(self.pixels):
                self.pixvals[i][j] = rgb[pixels[0],
                                         pixels[1], :] #designed to take all 3 color bands
        
                                 

