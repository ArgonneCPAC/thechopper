import numpy as np
import h5py
import glob
import os
from mpi4py import MPI
from thechopper.data_chopper import get_data_for_rank
from scipy.spatial import cKDTree

basepath = '/homes/avillarreal/repositories/thechopper/notebooks'
nx, ny, nz = 2, 2, 2
period, rmax = 1000, 30

comm = MPI.COMM_WORLD
rank, nranks = comm.Get_rank(), comm.Get_size()

all_sample0_fnames = sorted(
    glob.glob(basepath+"/DATA/sample0*.h5"))
all_sample1_fnames = sorted(
    glob.glob(basepath+"/DATA/sample1*.h5"))
#print('files to read in sample0: {}'.format(all_sample0_fnames))

sample0_len = len(all_sample0_fnames)
sample1_len = len(all_sample1_fnames)
rankbreak_sample0 = sample0_len / nranks
rankbreak_sample1 = sample1_len / nranks

lowrank = rank * rankbreak_sample0
highrank = (rank+1) * rankbreak_sample0
num_data = 0
i = 0
master_data0 = dict()
master_data1 = dict()
for fname0, fname1 in zip(all_sample0_fnames, all_sample1_fnames):
    if (highrank > i >= lowrank):
        #print('rank {} reading {}'.format(rank, fname0))
        with h5py.File(fname0, 'r') as hdf0:
            data0 = dict()
            data0['x'] = hdf0['x'][...].astype('f4')
            data0['y'] = hdf0['y'][...].astype('f4')
            data0['z'] = hdf0['z'][...].astype('f4')
            data0['mass'] = hdf0['mass'][...].astype('f4')
            data0['core_id'] = hdf0['core_id'][...].astype('i8')

        with h5py.File(fname1, 'r') as hdf1:
            data1 = dict()
            data1['x'] = hdf1['x'][...].astype('f4')
            data1['y'] = hdf1['y'][...].astype('f4')
            data1['z'] = hdf1['z'][...].astype('f4')
            data1['mass'] = hdf1['mass'][...].astype('f4')
            data1['core_id'] = hdf1['core_id'][...].astype('i8')
        if master_data0.keys():
            for key in master_data0.keys():
                master_data0[key] = np.append(master_data0[key],data0[key])
        else:
            for key in data0.keys():
                master_data0[key] = data0[key]
        if master_data1.keys():
            for key in master_data1.keys():
                master_data1[key] = np.append(master_data1[key],data1[key])
        else:
            for key in data1.keys():
                master_data1[key] = data1[key]
    i+=1
print('rank {} read in {} points'.format(rank, len(master_data0['x'])))
num_read = comm.gather(len(master_data0['x']), root=0)
if rank == 0:
    print('total read points is {}'.format(np.sum(num_read)))
data0_for_rank = get_data_for_rank(comm, master_data0, nx, ny, nz, period, 0)
data1_for_rank = get_data_for_rank(comm, master_data1, nx, ny, nz, period, rmax)
for subvol0_id, subvol0_data in data0_for_rank.items():
    sample0 = data0_for_rank[subvol0_id]
    num_data += len(sample0['x'])
print('rank {} has {} points'.format(rank, num_data))
num_chopped = comm.gather(num_data, root=0)
if rank == 0:
    print('total chopped points is {}'.format(np.sum(num_chopped)))
rank_counts = 0
# now that data is read, let's try doing pair counting
for subvol0_id, subvol0_data in data0_for_rank.items():
    sample0 = data0_for_rank[subvol0_id]
    mask0 = sample0['_inside_subvol'] == True
    x0 = sample0['x'][mask0]
    y0 = sample0['y'][mask0]
    z0 = sample0['z'][mask0]
    pos0 = np.vstack((x0, y0, z0)).T
    tree0 = cKDTree(pos0)
    try:
        sample1 = data1_for_rank[subvol0_id]
        x1 = sample1['x']
        y1 = sample1['y']
        z1 = sample1['z']
        pos1 = np.vstack((x1, y1, z1)).T
        tree1 = cKDTree(pos1)
        these_counts = tree0.count_neighbors(tree1, rmax)
    except KeyError:
        these_counts = 0
    rank_counts += these_counts
total_counts = comm.gather(rank_counts, root = 0)
if rank == 0:
    print('total pairs counted = {}'.format(np.sum(total_counts)))
