__author__ = "Mohit Sharma"

# No Interactive Plotting/ Plotting on screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import numpy as np
from itertools import chain

APP_NAME = "Hadoop_Image_Cluster"

class HadoopImageCluster(object):
    """
    Hadoop Image Cluster
    """
    def __init__(self, sc, path=None, fname=None, fname_ext=None, n=4, rows=2160, cols=4096, dims=3):
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
        conf = conf.set("spark.executor.memory", "12g")
        conf = conf.set("spark.executor.cores", "30")
        conf = conf.set("spark.driver.memory", "12g")
        # To solve for losing spark executors
        conf = conf.set("spark.network.timeout", "36000000")
        conf = conf.set("spark.yarn.executor.memoryOverhead", "6000")
        conf = conf.set("spark.dynamicAllocation.enabled", "true")
        conf = conf.set("spark.executor.heartbeatInterval", "36000000")
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
                                       n = self.n,
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
            imgpair = self.sc.binaryFiles(os.path.join(path, fname), 
                                          minPartitions=100)
        else:
            # /<path>/*.raw
            imgpair = self.sc.binaryFiles(os.path.join(path, "*"+fname_ext), 
                                          minPartitions=100)
        
        def _reshape(x):
            return [x[i*img_size: (i+1)*img_size].\
                        reshape(nrows, ncols, ndims) \
                        for i in range(n)]
    
        rdd = imgpair.map(lambda (x,y): (x, (np.asarray(bytearray(y), 
                                                        dtype=np.uint8))))
        
        rdd_resh = rdd.mapValues(_reshape)

        return rdd_resh


    def mean(self, n, bin, asdf=True):
        """
        Return the mean of the n-dim numpy array
        n: int
            if arrays are vertically stacked, n is the 
            total number of stacked arrays
        bin: int
            factor to bin the image by
        asdf: bool
            True: return the output as a dataframe
                To see the output, df.show(truncate=False)
            False: return the output as RDD
                To see the output, rdd.collect()

        """
        mean_rdd = self.img_rdd.mapValues(lambda x: \
                                              [x[k][::bin, ::bin].mean(axis=(0,1))\
                                                   for k in range(n)])

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

    def std(self, n, bin, asdf=True):
        """
        Return standard deviation for images
        Parameters
        ----------
        n: int
            if arrays are vertically stacked, n is the 
            total number of stacked arrays
        bin: int
            factor to bin the image by
        asdf: bool
            True: return the output as a dataframe
            To see the output, df.show(truncate=False)
            False: return the output as RDD
            To see the output, rdd.collect()

        """
        std_rdd = self.img_rdd.mapValues(lambda x: [x[k][::bin, ::bin].std()\
                                                        for k in range(n)])
        
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

    def getBright(self, n, bin):
        """
        Find the pixels brighter than 5 sigma
        Parameters
        ----------
        n: int
            n for n*sigma outlier
        bin: int
            factor to bin the image by
        .. note: Seriously need to come up with better names
        """
        def _getbright(x):
            result = []
            for k in range(combined):
                result.append((x[k][::bin,::bin] > \
                                   (x[k][::bin,::bin].mean(axis=(0,1))+ \
                                        n*(x[k][::bin,::bin].std()))).\
                                  any(-1).sum())
            return result
        
        bright = self.img_rdd.mapValues(_getbright)
        return bright

if __name__ == "__main__":
#    f_path = '/user/mohitsharma44/uo_images/bad_combined'
#    f_path = '/home/cusp/gdobler/cuip/cuip/hadoop/output/combined_images'
    inpath   = '/user/mohitsharma44/input/combined_bin8/'
    outpath  = '/user/mohitsharma44/output/'
    file_ext = ".raw"
    binfac   = 8
    nrows    = 2160 // binfac
    ncols    = 4096 // binfac
    ndims    = 3
    combined = 4 * binfac * binfac
    hic = HadoopImageCluster(sc=None, 
                             path=inpath, 
                             fname=None, 
                             fname_ext=file_ext, 
                             n=combined, 
                             rows=nrows, 
                             cols=ncols, 
                             dims=ndims)
    # Obtain Mean
    #df = hic.mean(combined, nrows=rows, ncols=cols, ndims=dims, asdf=True)
    #df.show(truncate = False)
    # Obtain Std dev
    #std = hic.std(combined, nrows=rows, ncols=cols, ndims=dims, asdf=False)
    res = hic.getBright(5, 1)
    def toStr(data):
        return ','.join(str(d) for d in data)
    out = res.map(toStr)
    out.saveAsTextFile(os.path.join(outpath, "dataplot.txt_bin_8"))
