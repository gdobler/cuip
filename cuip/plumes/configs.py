import os
MAXPROCESSES = 5e6
OUTPUTDIR = "./outputs/"
LIM = -1 #25
WINDOW = 5
try:
    DST_WRITE = os.environ['DST_WRITE']
except KeyError:
    DST_WRITE = os.environ['HOME']

