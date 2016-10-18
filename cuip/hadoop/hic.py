__author__ = "Mohit Sharma"

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import numpy as np
import os
from itertools import chain

APP_NAME = "Hadoop_Image_Cluster"

class HadoopImageCluster(object):
    """
    Hadoop Image Cluster
    """
    def __init__(self, sc, path=None, fname=None, fname_ext=None, n, rows, cols, dims):
        """
        Parameters
        ----------
        sc: `SparkContext`, optional
        path: str
            Location in hdfs containing files
        example: `/user/<username>/<dir>`
        fname: str, optional
            filename to be processed
        fname_ext: str
            extension for binary filenames
        n: int
            if arrays are vertically stacked, n is the 
            total number of stacked arrays
        nrows: int
            number of rows (per stacked array)
        ncols: int
            number of columns (per stacked array)
        ndims: int
            `ndims` dimensions

        Returns:
            img_rdd: rdd dataset containing images as numpy arrays
            reshaped as `(rows, cols, dims)`
        """
        conf = SparkConf().setAppName(APP_NAME)#.setMaster('local[*]')
        conf = conf.set("spark.executor.memory", "2g")
        conf = conf.set("spark.executor.cores", "2")
        if sc:
            self.sc = sc
        else:
            self.sc = SparkContext(conf=conf)
        self.sqlcontext = SQLContext(self.sc)
        self.path = path
        self.fname = fname
        self.fname_ext = fname_ext
        self.n = n
        self.rows = rows
        self.cols = cols
        self.dims = dims

        self.img_rdd = self._getImgRDD(self.path, 
                                       self.fname, 
                                       self.fname_ext,
                                       nrows = self.rows,
                                       ncols = self.cols,
                                       ndims = self.dims)
        #self.img_rdd.persist()


    def _getImgRDD(self, path, fname, fname_ext, n, nrows, ncols, ndims):
        """
        Return rdd of binary files converted to 
        numpy arrays. The numpy arrays will be
        reshaped as `(nrows, ncols, ndims)`
        Parameters
        ----------
        path: str
            Location in hdfs containing files
            example: `/user/<username>/<dir>`                                                                                                                               
        fname: str, optional
            filename to be processed
        fname_ext: str
            extension for binary filenames
        n: int
            if arrays are vertically stacked, n is the 
            total number of stacked arrays
        nrows: int
            number of rows (per stacked array)
        ncols: int
            number of columns (per stacked array)
        ndims: int
            `ndims` dimensions

        """
        img_size = nrows*ncols*ndims

        if fname:
            imgpair = self.sc.binaryFiles(os.path.join(path, fname))
        else:
            # /<path>/*.raw
            imgpair = self.sc.binaryFiles(os.path.join(path, "*"+fname_ext))
        
        def _reshape(x):
            img_size = rows*cols*dims
            return [x[i*img_size: (i+1)*img_size].reshape(rows, cols, dims) for i in range(n)]
    
        rdd = imgpair.map(lambda (x,y): (x, (np.asarray(bytearray(y), 
                                                        dtype=np.uint8))))
        
        rdd_resh = rdd.mapValues(_reshape)

        return rdd_resh


    def mean(self, n, asdf=True):
        """
        Return the mean of the n-dim numpy array
        n: int
            if arrays are vertically stacked, n is the 
            total number of stacked arrays
        asdf: bool
            True: return the output as a dataframe
                To see the output, df.show(truncate=False)
            False: return the output as RDD
                To see the output, rdd.collect()

        """
        mean_rdd = img_rdd.mapValues(lambda x: [x[k].mean(axis=(0,1)) for k in range(n)])

        if not asdf:
            return mean_rdd
        else:
            # --- Correct the output output of DF!!
            mean_formatted = mean_rdd.map(lambda x: (os.path.basename(x[0]), 
                                                     float(x[1][0][1]), 
                                                     float(x[1][1][1]), 
                                                     float(x[1][2][1])
                                                     ))
            # Dataframe Structure
            schema = StructType([StructField("Filename", StringType(), True), 
                                 StructField("CH0", DoubleType(), True), 
                                 StructField("CH1", DoubleType(), True), 
                                 StructField("CH2", DoubleType(), True)
                                 ])
            
            df = self.sqlcontext.createDataFrame(mean_formatted, schema)
            return df

    def std(self, n, asdf=True):
        """
        Return standard deviation for images
        Parameters
        ----------
        n: int
            if arrays are vertically stacked, n is the 
            total number of stacked arrays
        asdf: bool
            True: return the output as a dataframe
            To see the output, df.show(truncate=False)
            False: return the output as RDD
            To see the output, rdd.collect()

        """
        std_rdd = img_rdd.mapValues(lambda x: [x[k].std() for k in range(n)])
        
        if not asdf:
            return std_rdd
        else:
            std_formatted = std_rdd.map(lambda x: (os.path.basename(x[0]),
                                                     float(x[1]),
                                                     ))
            # Dataframe Structure                                                                                                                                                    
            schema = StructType([StructField("Filename", StringType(), True),
                                 StructField("STD", DoubleType(), True)
                                 ])

            df = self.sqlcontext.createDataFrame(std_formatted, schema)
            return df

    def getBright(n, plot=True):
        """
        Find the pixels brighter than 5 sigma
        Parameters
        ----------
        n: int
            n for n*sigma outlier
        plot: bool
            save plot of brightpixels > n*sigma
        .. note: Seriously need to come up with better names
        """
        def _getbright(x):
            result = []
            for k in range(combined):
                result.append((x[k] > (x[k].mean(axis=(0,1))+ n*(x[k].std()))).any(-1).sum())
            return result
        
        bright = img_rdd.mapValues(_getbright)
        if plot = False:
            return bright
        if plot = True:
            brights = bright.toLocalIterator()
            merged = list(chain.from_iterable([x[1] for x in brights]))
            plt.plot([i for i in range(len(merged))], merged)
            
            
if __name__ == "__main__":
    f_path = '/user/mohitsharma44/uo_images'
    f_ext = '.raw'
    rows = 2160
    cols = 4096
    dims = 3
    combined = 4
    hic = HadoopImageCluster(sc=None, path=f_path, fname=None, fname_ext=f_ext, combined, nrows, ncols, ndims)
    # Obtain Mean
    #df = hic.mean(combined, nrows=rows, ncols=cols, ndims=dims, asdf=True)
    #df.show(truncate = False)
    # Obtain Std dev
    #std = hic.std(combined, nrows=rows, ncols=cols, ndims=dims, asdf=False)
    hic.getBright(5, True)
