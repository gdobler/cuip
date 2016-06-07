"""
detect_match.py
Author: Chris Prince [cmp670@nyu.edu]
Date: 25 May 2016
"""

import cv2
import numpy as np
import pylab as pl
import os

def drawMatches(img1, kp1, img2, kp2, matches):
    """
        Adapted from Ray Phan's code at
        http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python

        Implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated
        keypoints, as well as a list of DMatch data structure (matches)
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
    """
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')
    # Place the first image to the left
    out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
    # Place the next image to the right of it
    out[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2, img2])
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)),
                 (int(x2)+cols1, int(y2)), (255, 0, 0), 1)
    # Show the image
    return out


def feature_match(img1, img2, detectAlgo=cv2.SIFT(), matchAlgo='bf',
                  bfNorm=cv2.NORM_L2,
                  flannIndexParams=dict(algorithm=0,  # =cv2.FLANN_INDEX_KDTREE
                                        trees=5),
                  flannSearchParams=dict(checks=20)
                  ):
    """
    Find features in images img1 and img2 using a detection algorithm (default
    is cv2.SIFT), then match them according to a brute force ('bf') or fast
    approximation ('flann') method. Returns the features and the matches.
    """

    support_detect = ['sift', 'surf', 'orb']
    support_match = ['bf', 'flann']

    # The actual feature detection computations
    kp1, des1 = detectAlgo.detectAndCompute(img1, None)
    kp2, des2 = detectAlgo.detectAndCompute(img2, None)

    if matchAlgo not in support_match:
        errmsg = 'Matching algorithm {} not supported ' \
                 '(must be one of {})'.format(
                  matchAlgo, ', '.join(support_match))
        raise ValueError(errmsg)
    elif matchAlgo == 'bf':
        # bf = cv2.BFMatcher(cv2.NORM_L2) #cv2.NORM_HAMMING)
        # TODO: decide whether to allow norm to be chosen by user or make dependent on algorithm
        matcher = cv2.BFMatcher(bfNorm)
    elif matchAlgo == 'flann':
        # FLANN is an optimized matcher (versus brute force), so should play
        # with this as well. Maybe brute force is better if we go with multiple
        # subsamples to register image (instead of entire image)
        matcher = cv2.FlannBasedMatcher(flannIndexParams, flannSearchParams)
    else:
        print 'should not have gotten here!'
        raise Exception('wow')

    # The actual matching computation
    matches = matcher.match(des1, des2)

    # Return the keypoints and matches as a tuple
    return (kp1, des1, kp2, des2, matches)


def img_cdf(img):
    """
    Calculate the CDF of an image using quicksort.
    For this application we don't need the actual histograms.
    """

    x = np.sort(img, axis=None)
    dims = img.shape[:2]
    pixelcount = dims[0]*dims[1]
    f = np.arange(pixelcount)/float(pixelcount)

    # Return the cdf as a tuple of the sorted array and normalized index
    return (x, f)


def neighborhood_cdf(img):
    """
    Calculate the CDF of an image with additional weighting for points in
    each pixel's neighborhood--with any luck this allows for each pixel to
    be ranked with (very few) ties.

    Create a new matrix using the reference pixels as base and add up to a full
    intensity level depending on the intensities of the surrounding pixels:

         |x|
       |x|x|x|
     |x|x|O|x|x|
       |x|x|x|
         |x|

    """
    # TODO: break this out into its own routine with options for neighbor
    # definitions and weighting

    # Add 1/255th of the average of 12 nearby pixels to the original image
    im12 = img[2:-2, 2:-2, :] + (
            img[3:-1, 2:-2, :] + img[1:-3, 2:-2, :] +  # sides d=1
            img[2:-2, 1:-3, :] + img[2:-2, 3:-1, :] +
            img[3:-1, 3:-1, :] + img[3:-1, 1:-3, :] +  # corners d=1
            img[1:-3, 3:-1, :] + img[1:-3, 1:-3, :] +
            img[4:, 2:-2, :] + img[:-4, 2:-2, :] +     # sides d=2
            img[2:-2, 4:, :] + img[2:-2, :-4, :]) / 12./255.
    # no. of pixels / max 8-bit intensity

    return img_cdf(im12[:,:,0])


if __name__ == '__main__':
    # Here's my sample image; could replace with a command line option
    fname = os.getenv('cuipimg') + 'temp__2014-09-29-125314-29546.raw'
    fname2 = os.getenv('cuipimg') + 'temp__2016-03-16-114846-163394.raw'
    img = np.fromfile(fname, dtype=np.uint8)
    img = img.reshape(2160, 4096, 3)[:, :, ::-1]
    img2 = np.fromfile(fname2, dtype=np.uint8)
    img2 = img2.reshape(2160, 4096, 3)[:, :, ::-1]
    img2 = img[300:800, 300:800, :]
    # im4 = feature_match(img[300:800, 300:800, :], img) #, cv2.ORB())
    kp1, des1, kp2, des2, matches = feature_match(img2, img, cv2.SIFT(), 'flann')  # , cv2.ORB())
    im4 = drawMatches(img2[:,:,0], kp1, img[:,:,0], kp2, matches)
    pl.imshow(im4)
    # Assumes the Qt4Agg backend in place for matplotlib
    pl.show()

    xx = []
    yy = []
    dd = []
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        dx = x2-x1
        dy = y2-y1
        xx.append(dx)
        yy.append(dy)
        # Compute the distance between matching keypoints
        d = (dx**2 + dy**2)**0.5
        dd.append(d)

    pl.hist(dd)
    pl.show()

    print 'dx mean = {}, dx sd = {}'.format(np.mean(xx), np.std(xx))
    print 'dy mean = {}, dy sd = {}'.format(np.mean(yy), np.std(yy))
    print 'dist mean = {}, dist sd = {}'.format(np.mean(dd), np.std(dd))
