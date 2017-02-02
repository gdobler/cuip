#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

path_nobin = '/home/cusp/mohitsharma44/dataplot_nobin/'
path_2bin = '/home/cusp/mohitsharma44/dataplot_bin_2/'
path_4bin = '/home/cusp/mohitsharma44/dataplot_bin_4/'
path_8bin = '/home/cusp/mohitsharma44/dataplot_bin_8/'

def getPoints(path, binfac):
    files = [os.path.join(path, x) for x in os.listdir(path) if x.startswith('part')]
    all_points = []
    def _readfiles(path):
        for f in files:
            with open(f, 'r') as fh:
                all_points.append(fh.readlines())
        return all_points

    all_points = [x.split('/')[-1].strip(']\n').split('.raw,[') 
                  for data in _readfiles(path) for x in data]

    all_points = [dict(zip(range(int(x[0].split('_')[0]), int(x[0].split('_')[1]) ), 
                     map(lambda v: binfac*binfac*int(v), x[1].split(',')))) 
            for x in all_points]
    all_points = sorted(map(lambda x: sorted(x.items()), all_points))
    return [x for y in all_points for x in y]

all_points_nobin = getPoints(path_nobin, 1)
all_points_2bin  = getPoints(path_2bin, 2)
all_points_4bin  = getPoints(path_4bin, 4)
all_points_8bin  = getPoints(path_8bin, 8)

#points_nobin = np.hstack([x.values() for x in all_points_nobin])
#points_2bin  = np.hstack([x.values() for x in all_points_2bin])
#points_4bin  = np.hstack([x.values() for x in all_points_4bin])
#points_8bin  = np.hstack([x.values() for x in all_points_8bin])

#point_array_nobin = np.asarray(points_nobin, dtype=np.int).flatten()
#point_array_2bin = np.asarray(points_2bin, dtype=np.int).flatten()
#point_array_4bin = np.asarray(points_4bin, dtype=np.int).flatten()
#point_array_8bin = np.asarray(points_8bin, dtype=np.int).flatten()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(*zip(*all_points_nobin), label="no_bin")
ax.plot(*zip(*all_points_2bin),  label="bin_x2")
ax.plot(*zip(*all_points_4bin), label="bin_x4")
ax.plot(*zip(*all_points_8bin), label="bin_x8")

#ax.plot(range(points_nobin.shape[1]), points_nobin[0][:points_nobin.shape[1]], label="no_bin")
#ax.plot(range(points_2bin.shape[1]),  points_2bin[0][:points_2bin.shape[1]]*4, label="bin x2")
#ax.plot(range(points_4bin.shape[1]),  points_4bin[0][:points_4bin.shape[1]]*16, label="bin x4")
#ax.plot(range(points_8bin.shape[1]),  points_8bin[0][:points_8bin.shape[1]]*64, label="bin x8")

plt.legend(loc="upper left")
plt.savefig('current.png', dpi=fig.dpi)
plt.show()
"""
print len(point_array_nobin)
print len(point_array_2bin)
print len(point_array_4bin)
print len(point_array_8bin)
"""
