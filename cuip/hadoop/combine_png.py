import numpy as np
from cuip.cuip.utils.misc import getFiles
from cuip.cuip.utils import cuiplogger
from scipy.ndimage import imread
import os

PATH = '/uofs10tb_gpfs/roof/1/'
OUTPATH = '/home/cusp/mohitsharma44/uo/cuip/cuip/hadoop/output'
logger = cuiplogger.cuipLogger(loggername="COMBINE", tofile=False)

# File shape
nrows = 2160
ncols = 4096
ndims = 3

# Date ranges
start_date = '2016.03.09'
start_time = '23.55.00'
end_date = '2016.03.10'
end_time = '00.05.00'

# Number of files to combine
combine = 4

# Get all the files
def groupFiles(n, gf):
    """
    Return path for `n` files in a list
    Parameters
    ----------
    n: int
        number of files to return in a list
    gf: `generator`
        generator containing all the file paths

    Returns
    -------
    flist: list
        list of n file paths
    """
    flist =  []
    for i in range(n):
        flist.append(gf_gen.next())
    return flist

# Array to store the images
comb_imgs = np.zeros([nrows*combine, ncols, ndims], np.uint8)
gf_gen = getFiles(PATH, start_date, start_time, end_date, end_time)

def pngtoraw(img_arr, gfgen, n, outpath):
    """
    Convert png images to raw files
    Parameters
    ----------
    img_arr: np.array
        numpy array of the size of the image
        example:
            img_arr = np.zeros([nrows*n, ncols, ndims], np.uint8)
            where `n` is the total number of images to be combined
            by stacking vertically
    gfgen: `generator`
        generator containing file paths
        .. note: `len` of the list should be same as `n`
    n: int
        Number of files to stack together vertically
    outpath: str
        path where the file should be written to
    """
    for ind, imgs in enumerate(groupFiles(n, gfgen)):
        img_arr[nrows*ind: nrows*(ind+1), :, :] = imread(imgs, mode='RGB')
    # Rename the extension of the file to .raw.
    # outfile name is same as the n(th) filename.raw
    newfname = os.path.basename(imgs)[:-4]+".raw"
    logger.info("Writing: "+str(newfname))
    img_arr.tofile(os.path.join(outpath, newfname))

# Combine images
while True:
    try:
        pngtoraw(comb_imgs, gf_gen, combine, OUTPATH)
    except StopIteration as si:
        logger.warning("Stop Iteration")
        break
