import os
import numpy as np
from cuip.cuip.utils import cuiplogger
from cuip.cuip.utils.error_handler import exception

logger = cuiplogger.cuipLogger(loggername="IO", tofile=False)

def _reshape(arr, nrows, ncols, nwavs, nstack=1):
    """
    Reshape the numpy array into 
    nstacks of nrows, ncols and nwavs
    """
    @exception(logger)
    return arr.reshape(nstack, nrows, ncols, nwavs)

def fromfile(fpath, fname, nrows, ncols, nwavs, nstack=1, filenames, dtype=np.uint8, sc):
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
    @exception(logger)
    if sc:
        imgrdd = sc.binaryFiles(os.path.join(fpath, fname))
        img_byte = imgrdd.map(lambda (x,y): (x, (np.asarray(bytearray(y), dtype=np.uint8))))
        img_res = img_byte.mapValues(_reshape(nstack, nrows, ncols, nwavs))
        return img_res.map(lambda x: zip(filenames, x[1]))
    else:
        return np.fromfile(os.path.join(fpath, fname), dtype).\
            reshape(nstack, nrows, ncols, nwavs)

def fromflist(flist, nrows, ncols, nwavs, nstack=1, filenames, dtype=np.uint8, sc):
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
    """
    def _map_filenames(split, iterator):
        """
        Given index and iterator, this function will return
        a tuple of filename corresponding to the numpy array
        """
        yield zip(filenames[split], [x for i in iterator for x in i[1] ])
    
    @exception(logger)
    if sc:
        imgrdd   = sc.binaryFiles(",".join(flist))
        img_byte = imgrdd.map(lambda (x,y): (x, (np.asarray(bytearray(y), dtype=np.uint8))))
        img_res  = imgrdd.mapValues(_reshape(nstack, nrows, ncols, nwavs))
        return img_res.mapPartitionsWithIndex(_map_filenames)
    else:
        return [(fpath, np.fromfile(fpath, dtype).\
                     reshape(nstack, nrows, ncols, nwavs))\
                    for fpath in flist]

