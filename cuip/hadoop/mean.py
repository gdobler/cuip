__author__ = "Mohit Sharma"

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import numpy as np

APP_NAME = "Image_Mean"

def mean(sc, rdd, stacked, n, nrows, ncols, ndims):
    """
    Obtain mean of the image
    Parameters
    ----------
    sc: `SparkContext`
    rdd: `pyspark.rdd` 
        dataset containing binary raw file
        example: rdd = sc.binaryFiles('/path/to/raw/file')
    stacked: bool
        If the rdd contains stacked images (to improve namenode efficiency)
    n: int
        if stacked == `True`, n is the number of stacked images
    nrows: int
        number of rows per image (per stacked image)
    ncols: int
        number of columns per image (per stacked image)
    ndims: int
        number of dimensions per image (eg. 3 for RGB image)
    """
    # Collect the data to create a numpy array.
    # -- Research a better way -- #
    img_rdd = rdd.collect()
    img_series = np.asarray(bytearray(img_rdd[0][1]), dtype=np.uint8)
    data = []

    def _mean(x): 
        mean1 = np.mean(x[:,:,0])
        mean2 = np.mean(x[:,:,1])
        mean3 = np.mean(x[:,:,2])
        return ('ch0',mean1), ('ch1', mean2), ('ch3', mean3)

    # For n stacked images, extract individual image and tag them
    for i in range(n):
        data.append(
            ('img'+str(i), 
             img_series[i*nrows*ncols*ndims : (i+1)*nrows*ncols*ndims].reshape(nrows, ncols, ndims)
             ))
    
    # Distribute the data to all the nodes
    rdd = sc.parallelize(data)
    # Map mean function on every image
    return rdd.mapValues(_mean)
    

if __name__ == "__main__":
    # Configure OPTIONS
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.set("spark.executor.memory", "2g")
    conf = conf.set("spark.executor.cores", "10")
    sc = SparkContext(conf=conf)
    sqlcontext = SQLContext(sc)

    ## Logger
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel( logger.Level.OFF )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.OFF )
    logger = logger.LogManager.getLogger(__name__)
    logger.info("Obtain Mean intensity of the pixel in every channel")
    
    # Obtain Mean        
    fname = '/user/mohitsharma44/uo_images/*.raw'
    imrdd = sc.binaryFiles(fname)
    means = mean(sc, rdd=imrdd, stacked=True, n=4, nrows=2160, ncols=4096, ndims=3)

    means_converted = means.map(lambda x: (x[0], 
                                           float(x[1][0][1]), 
                                           float(x[1][1][1]), 
                                           float(x[1][2][1])
                                           ))
    schema = StructType([StructField("IMG", StringType(), True), 
                         StructField("CH0", DoubleType(), True), 
                         StructField("CH1", DoubleType(), True), 
                         StructField("CH2", DoubleType(), True)
                         ])
    df = sqlcontext.createDataFrame(means_converted, schema)
    # Write it to a file
    # df.write.csv('mycsv.csv')
    logger.info("Files from: " +str(fname))
    logger.info("Means: "+str(df.show()))
