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

    def __new__(cls, img_array, comment=None, metadata=None ):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(img_array).view(cls)
        # add the new attribute to the created instance
        obj.comment = comment
        obj.metadata = metadata
        
        # Finally, we must return the newly created object:
        return obj
 
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.comment = getattr(obj, 'comment', None)
        self.metadata = getattr(obj, 'metadata', None)

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
            # read raw images and add to a tuple with fpath
            img = (os.path.join(fpath,fname), 
                    np.fromfile(os.path.join(fpath, fname), dtype).\
                        reshape(nstack, nrows, ncols, nwavs))
            # create dictionary of filenames with f_number as the key
            filenames = {filenames[0]: [filenames[1]]}
            # create list of fname and gname mapped images 
            fn_mapped = zip(repeat(img[0]), 
                            filenames[os.path.basename(img[0]).strip('.raw')],
                            img[1])
            print fn_mapped[0][0]
            print fn_mapped[0][1]
            # return a CuipImageArray with metadata containing gname and fname
            return [CuipImageArray(img_array=img[2], metadata={'gname': img[0],
                                                               'fname': img[1]})\
                        for img in fn_mapped]

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
            # create a list of all the files as a tuple of fname and ndarray
            img_list = [(fpath, np.fromfile(fpath, dtype).\
                             reshape(nstack, nrows, ncols, nwavs))\
                            for fpath in flist]
            # create dictionary of filenames with f_range as key
            filenames = dict((f_rng, fnames) for f_rng,fnames in filenames)
            fn_mapped = []
            # create list of fname and gname mapped images
            for img in img_list:
                fn_mapped.append(zip(repeat(img[0]), 
                                     filenames[os.path.basename(img[0]).strip(".raw")],
                                     img[1]))
            # return CuipImageArray for all the ndarrays.. flattened
            return [CuipImageArray(img_array=img[2], 
                                   metadata={"gname": img[0], 
                                         "fname": img[1]}) \
                        for sublist in fn_mapped for img in sublist]
            
    except Exception as ex:
        logger.error("Error processing flist: "+str(ex))
