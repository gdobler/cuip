from cuip.cuip.fileio import fromflist, fromfile
import numpy as np
import os
import csv

with open('/home/cusp/mohitsharma44/devel/uo/cuip/cuip/hadoop/2017-02-10-11-02-38.csv', 'r') as fh:
    csv_read = csv.DictReader(fh)
    csv_dict = [x for x in csv_read]

sorted_csv_dict = sorted(csv_dict, key=lambda x: int(x['test_uo_files_fnumber']))

files = os.listdir('/home/cusp/mohitsharma44/devel/uo/cuip/cuip/hadoop/output/combined_images')
files = [os.path.join('/home/cusp/mohitsharma44/devel/uo/cuip/cuip/hadoop/output/combined_images', f) for f in files if f.endswith(".raw")]

filenames = []
for f in files:
    start = int(os.path.basename(f).split('_')[0])
    end   = int(os.path.basename(f).split('_')[1].strip('.raw'))
    filenames.append(('{0}_{1}'.format(start, end), 
                      [os.path.join(x['test_uo_files_fpath'], 
                                    x['test_uo_files_fname']) for x in csv_dict \
                           if int(x['test_uo_files_fnumber']) in range(start, end)]))

binfac = 2
nrows  = 2160 // 2
ncols  = 4096 // 2
nwavs  = 3
nstack = 4 * binfac * binfac

ff = fromflist(files, nrows, ncols, nwavs, filenames, nstack, np.uint8, None)
f  = fromfile(os.path.dirname(files[0]), os.path.basename(files[0]), 
              nrows, ncols, nwavs, (filenames[0][0], filenames[0][1][0]), 
              nstack, np.uint8, None)
