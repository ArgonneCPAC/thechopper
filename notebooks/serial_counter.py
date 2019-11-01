"""
"""
import numpy as np
from scipy.spatial import cKDTree
import h5py
import glob
import os

rmax = 30
sample0_fnames = sorted(
    glob.glob("/Users/aphearin/work/random/1101/DATA/sample0*.h5"))
sample1_fnames = sorted(
    glob.glob("/Users/aphearin/work/random/1101/DATA/sample1*.h5"))

counts = 0
for fname0, fname1 in zip(sample0_fnames, sample1_fnames):
    with h5py.File(fname0, 'r') as hdf:
        x0 = hdf['x'][...]
        y0 = hdf['y'][...]
        z0 = hdf['z'][...]
    with h5py.File(fname1, 'r') as hdf:
        x1 = hdf['x'][...]
        y1 = hdf['y'][...]
        z1 = hdf['z'][...]
    pos0 = np.vstack((x0, y0, z0)).T
    pos1 = np.vstack((x1, y1, z1)).T
    tree0 = cKDTree(pos0, boxsize=1000)
    tree1 = cKDTree(pos1, boxsize=1000)
    these_counts = tree0.count_neighbors(tree1, rmax)
    print("For rank {0} filenum {1}, counts = {2}".format('0',
        os.path.basename(fname0).split('.')[0].split('_')[-1], these_counts))
    counts += these_counts

print("Total counts = {0}".format(counts))

