#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

path_nobin = '/home/cusp/mohitsharma44/dataplot_bin_1/'
path_2bin = '/home/cusp/mohitsharma44/dataplot_bin_2/'
path_4bin = '/home/cusp/mohitsharma44/dataplot_bin_4/'
path_8bin = '/home/cusp/mohitsharma44/dataplot_bin_8/'

def getPoints(path):
    files = [os.path.join(path, x) for x in os.listdir(path) if x.startswith('part')]
    all_points = []
    for f in files:
        with open(f, 'r') as fh:
            all_points.append(fh.readlines())

    all_points = map(lambda x: x.split('/')[-1].strip(']\n').split(',['), 
              [x for data in all_points for x in data])

    return sorted(all_points, key=itemgetter(0))

all_points_nobin = getPoints(path_nobin)
all_points_2bin = getPoints(path_2bin)
all_points_4bin = getPoints(path_4bin)
all_points_8bin = getPoints(path_8bin)

points_nobin = map(lambda x: x[1].split(', '), all_points_nobin)
points_2bin = map(lambda x: x[1].split(', '), all_points_2bin)
points_4bin = map(lambda x: x[1].split(', '), all_points_4bin)
points_8bin = map(lambda x: x[1].split(', '), all_points_8bin)

point_array_nobin = np.asarray(points_nobin, dtype=np.int).flatten()
point_array_2bin = np.asarray(points_2bin, dtype=np.int).flatten()
point_array_4bin = np.asarray(points_4bin, dtype=np.int).flatten()
point_array_8bin = np.asarray(points_8bin, dtype=np.int).flatten()

fig = plt.figure()
ax = fig.add_subplot(111)


ax.plot(range(len(point_array_nobin)), point_array_nobin, label="no_bin")
ax.plot(range(len(point_array_2bin)), point_array_2bin*4, label="bin x2")
ax.plot(range(len(point_array_4bin)), point_array_4bin*16, label="bin x4")
ax.plot(range(len(point_array_8bin)), point_array_8bin*64, label="bin x8")
plt.legend(loc="upper left")
#plt.savefig('current.png', dpi=fig.dpi)
plt.show()
"""
print len(point_array_nobin)
print len(point_array_2bin)
print len(point_array_4bin)
print len(point_array_8bin)
"""
