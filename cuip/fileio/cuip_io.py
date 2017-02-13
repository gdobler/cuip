import os
import numpy as np
from operator import itemgetter
from itertools import repeat
from cuip.cuip.utils import cuiplogger
logger = cuiplogger.cuipLogger(loggername="IO", tofile=False)

import numpy as np

class CuipImageArray(np.ndarray):
    """
    Parameters
    ----------
    input_array: np.ndarray
        stacked numpy array. example
        an 'rgb' image array of 2160 x 4096 resolution
        with 10 stacked images 
        will have shape of shape = (10, 2160, 4096, 3)
    comment: optional
        any python datatype. preferably a string.
    """

    def __new__(cls, input_array, comment=None, filenames=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.comment = comment
        
        
        # Finally, we must return the newly created object:
        return obj
 
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.comment = getattr(obj, 'comment', None)

def _reshape(arr, nrows, ncols, nwavs, nstack=1):
    """
    Reshape the numpy array into 
    nstacks of nrows, ncols and nwavs
    """
    try:
        reshaped_img = arr.reshape(nstack, nrows, ncols, nwavs)
        return reshaped_img
    except Exception as ex:
        logger.error("Error reshaping the array: "+str(ex))

def fromfile(fpath, fname, nrows, ncols, nwavs, filenames, nstack, dtype, sc):
    """
    Read image file to a rdd as binary file if sc is passed
    else return the file content as a numpy array
    
    Parameters
    ----------
    fpath: str
        file directory path
    fname: str
        file name with extension
        note..: currently only raw is supported
    nrows, ncols, nwavs: int
        shape of the image
    nstack: int
        If passed, will cause single files to be 
        subdivided into nstack separate images.
    filenames: 1-d array or list
        If nstack > 1, this optional parameter
        can set the rdd as tuple of (filename, np.array)
    dtype: data type of the image
        default: numpy.uint8
    sc: sparkContext
        if calling the function in spark cluster, this
        function will return a RDD of binary file
    Returns
    -------
    numpy.array OR spark RDD
    """
    try:
        if sc:
            imgrdd = sc.binaryFiles(os.path.join(fpath, fname))
            img_byte = imgrdd.map(lambda (x,y): (x, (np.asarray(bytearray(y), dtype=np.uint8))))
            img_res = img_byte.flatMap(lambda x: _reshape(x[1], nstack, nrows, ncols, nwavs))
            return img_res.map(lambda x: zip(filenames, x[1]))
        else:
            img = np.fromfile(os.path.join(fpath, fname), dtype).\
                reshape(nstack, nrows, ncols, nwavs)
            return zip(filenames, img)
    except Exception as ex:
        logger.error("Error loading file: "+str(ex))

def fromflist(flist, nrows, ncols, nwavs, filenames, nstack, dtype, sc):
    """
    Read files from a list as a binary file if sc is passed
    else return a list of tuple with filename and the binary
    content as a numpy array.
    
    Parameters
    ----------
    flist: list
        list of files to be read from
    nrows, ncols, nwavs: int
        shape of the image
    nstack: int
        It will cause single files to be 
        subdivided into nstack separate images.
    filenames: 1-d array or list of `list of filenames`
        If nstack > 1, this optional parameter
        can set the rdd as tuple of (filename, np.array)
    dtype: data type of the image
        default: np.uint8
    sc: sparkContext
        if calling the function in spark cluster, this
        function will return a RDD of all files

    Returns
    -------
    if sc:
        return RDD which will be in format ()
    else:
        return list of 
    """
    def _map_filenames(split, iterator):
        """
        Given index and iterator, this function will return
        a tuple of filename corresponding to the numpy array
        """
        yield zip(filenames[split], [x for i in iterator for x in i[1] ])
    try:
        if sc:
            imgrdd   = sc.binaryFiles(",".join(flist))
            img_byte = imgrdd.map(lambda (x,y): (x, (np.asarray(bytearray(y), dtype=np.uint8))))
            img_res  = imgrdd.flatMap(lambda x: _reshape(x[1], nstack, nrows, ncols, nwavs))
            return img_res.mapPartitionsWithIndex(_map_filenames)
        else:
            img_list = [(fpath, np.fromfile(fpath, dtype).\
                             reshape(nstack, nrows, ncols, nwavs))\
                            for fpath in flist]
            filenames = dict((f_rng, fnames) for f_rng,fnames in filenames)
            final = []
            for img in img_list:
                final.append(zip(repeat(img[0]), 
                                 img[1], 
                                 filenames[os.path.basename(img[0]).strip(".raw")]))
            return [img for sublist in final for img in sublist]
            
            """
            img_list = [(fpath, np.fromfile(fpath, dtype).\
                            reshape(nstack, nrows, ncols, nwavs))\
                            for fpath in flist]
            # ToDo: Optimize this ..
            return [(x[0], zip(fnames[1], x[1])) for x in img_list for fnames in filenames if fnames[0] in os.path.basename(x[0])]
            #return [[(x[0], zip(fnames[1], x[1])) for fnames in filenames if fnames[0] in os.path.basename(x[0])][0] for x in img_list]
            """
    except Exception as ex:
        logger.error("Error processing flist: "+str(ex))
