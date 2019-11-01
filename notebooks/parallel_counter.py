"""
mpiexec -n 2 python parallel_counter.py

"""
import numpy as np
from scipy.spatial import cKDTree
import h5py
import glob
import os
from mpi4py import MPI
from thechopper.data_chopper import get_data_for_rank

period, rmax = 1000, 30
nx, ny, nz = 2, 2, 2

comm = MPI.COMM_WORLD
rank, nranks = comm.Get_rank(), comm.Get_size()

all_sample0_fnames = sorted(
    glob.glob("/Users/aphearin/work/random/1101/DATA/sample0*.h5"))
all_sample1_fnames = sorted(
    glob.glob("/Users/aphearin/work/random/1101/DATA/sample1*.h5"))

# sample0_fnames = np.array_split(all_sample0_fnames, nranks)
# sample1_fnames = np.array_split(all_sample1_fnames, nranks)

rank_counts = 0
for fname0, fname1 in zip(all_sample0_fnames, all_sample1_fnames):

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

    data0_for_rank = get_data_for_rank(comm, data0, nx, ny, nz, period, 0)
    data1_for_rank = get_data_for_rank(comm, data1, nx, ny, nz, period, rmax)

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

        # print("subvol0_id = {0}".format(subvol0_id))
        outpat = "DATA/data{0}_subvolID_{1}_for_rank_{2}.h5"
        if os.path.basename(fname0).split('.')[0].split('_')[-1] == '3':
            outname = outpat.format('0', subvol0_id, rank)
            # print("outname = {}".format(outname))
            with h5py.File(outname, 'w') as hdf:
                for key in sample0.keys():
                    hdf[key] = sample0[key]
            outname = outpat.format('1', subvol0_id, rank)
            with h5py.File(outname, 'w') as hdf:
                for key in sample1.keys():
                    hdf[key] = sample1[key]
    print("rank = {0} filenum {1}, counts = {2}".format(rank,
        os.path.basename(fname0).split('.')[0].split('_')[-1], these_counts))
    comm.Barrier()


print("rank = {0}\nrank_counts = {1}".format(rank, rank_counts))





