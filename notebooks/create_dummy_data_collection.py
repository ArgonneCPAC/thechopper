"""
"""
import numpy as np
import h5py
from thechopper.dummy_data import dummy_halo_properties


num_readers = 2
num_files = 6

counter = 0
for ifile in range(num_files):
    npts = np.random.randint(5000, 15000)
    smin, smax = np.sort(np.random.uniform(0, 1000, 2))
    # smin, smax = 0, 1000
    data, metadata = dummy_halo_properties(npts, num_readers, smax,
            seed=ifile, box_min=smin, core_id_min=counter)
    counter += npts

    outname = "DATA/sample0_dummy_data_{0}.h5".format(ifile)
    with h5py.File(outname, 'w') as hdf:
        for key in data.keys():
            hdf[key] = data[key]

for ifile in range(num_files):
    npts = np.random.randint(500, 1500)
    smin, smax = np.sort(np.random.uniform(0, 1000, 2))
    data, metadata = dummy_halo_properties(npts, num_readers, smax,
            seed=ifile+counter, box_min=smin, core_id_min=counter)
    counter += npts

    outname = "DATA/sample1_dummy_data_{0}.h5".format(ifile)
    with h5py.File(outname, 'w') as hdf:
        for key in data.keys():
            hdf[key] = data[key]
