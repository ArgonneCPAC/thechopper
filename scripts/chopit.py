"""
Example script demonstrating how to partition data into buffered subvolumes and
distribute to MPI ranks for parallel processing.

mpiexec -n 2 python chopit.py 2 2 2 1000 30 mass core_id
"""
import argparse
from mpi4py import MPI
import numpy as np
from thechopper.dummy_data import dummy_halo_properties
from thechopper import get_chopped_data, assign_chopped_data_to_ranks


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("nx_divs", type=int, help="Number of divisions in the x-dimension")
    parser.add_argument("ny_divs", type=int, help="Number of divisions in the y-dimension")
    parser.add_argument("nz_divs", type=int, help="Number of divisions in the z-dimension")
    parser.add_argument("period", type=float, help="Size of the periodic box")
    parser.add_argument("rmax", type=float, help="Maximum search length. Defines the size of the subvolume buffer.")
    parser.add_argument("columns_to_retrieve", help="Name of columns to retrieve", nargs='*')

    args = parser.parse_args()

    nx_divs, ny_divs, nz_divs = args.nx_divs, args.ny_divs, args.nz_divs
    period = args.period
    rmax = args.rmax
    columns_to_retrieve = args.columns_to_retrieve

    #  Fire up a communicator with one rank per compute node
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    rng = np.random.RandomState(rank)

    #  For each rank, generate some dummy data that mimics
    #  the chunk of data the rank would read from disk.
    num_dummy_data_points = rng.randint(500, 1000)
    new_data, new_metadata = dummy_halo_properties(num_dummy_data_points, nranks, period)
    #  In this setup, the initial data processed by each rank could be anywhere in the box
    #  The end goal will be for each rank to end up with only data points in the
    #  buffered subvolume(s) assigned to the rank

    #  Get to the choppa!
    chopped_data = get_chopped_data(
        new_data, nx_divs, ny_divs, nz_divs, period, rmax, columns_to_retrieve)
    #  chopped_data is a dictionary. The keys are names of halo properties.
    #  A list of length nx_divs*ny_divs*nz_divs will be bound to each key.
    #  Each element in the list is a (possibly empty) ndarray

    #  Now that data has been partitioned into buffered subvolumes,
    #  we assign each rank one or more subvolumes
    flattened_chopped_data, npts_to_send_to_rank = assign_chopped_data_to_ranks(
        chopped_data, nx_divs, ny_divs, nz_divs, nranks)
    #  flattened_chopped_data is the same as chopped_data,
    #  but for each column, the list of ndarrays has been concatenated
    #  npts_to_send_to_rank is an ndarray of length nranks

    #  Now we calculate npts_to_receive_from_rank by using
    #  the MPI Alltoall function to transpose the collection of npts_to_send_to_rank
    npts_to_receive_from_rank = np.empty_like(npts_to_send_to_rank)
    comm.Alltoall(npts_to_send_to_rank, npts_to_receive_from_rank)
    comm.Barrier()
    #  Now every rank knows how many points to send to each rank (npts_to_send_to_rank)
    #  and also how many points to receive from each rank (npts_to_receive_from_rank)
    #  This is all we need to use Alltoallv to distribute the data between ranks

    data_for_rank = dict()
    for colname in flattened_chopped_data.keys():
        sendbuf = flattened_chopped_data[colname]
        send_counts = npts_to_send_to_rank

        recv_counts = npts_to_receive_from_rank
        recv_buff = np.empty(recv_counts.sum(), dtype=sendbuf.dtype)
        comm.Alltoallv([sendbuf, send_counts], [recv_buff, recv_counts])
        comm.Barrier()

        data_for_rank[colname] = recv_buff

    msg = "\nFor rank {0}, xmin, xmax = ({1}, {2})"
    print(msg.format(rank, data_for_rank['x'].min(), data_for_rank['x'].max()))

    msg = "\nFor rank {0}, ymin, ymax = ({1}, {2})"
    print(msg.format(rank, data_for_rank['y'].min(), data_for_rank['y'].max()))
