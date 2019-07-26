"""Module implements generate_chopped_data that yields a data stream of xyz points
that have been run through the chopper and distributed across MPI ranks.
"""
import numpy as np
from .buffered_subvolume_calculations import points_in_buffered_rectangle, subvol_bounds_generator
from .buffered_subvolume_calculations import rectangular_subvolume_cellnum


def generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax):
    """Generator using thechopper to distribute xyz data across MPI ranks.

    Parameters
    ----------
    x, y, z : ndarrays
        Arrays with shape (npts, ) storing Cartesian coordinates of the points being chopped

    rank : int

    nranks : int

    nx, ny, nz : ints
        Number of divisions of the box in each dimension

    period : float or 3-element sequence
        Periodic boundary conditions in each dimension.
        Generator assumes the data volume is a cube if passed a float for period.

    rmax : float or 3-element sequence
        Maximum search length required by any of the summary statistics.

    Returns
    -------
    subvol_indx : int
        Cell ID of the subvolume of data being yielded to the input rank

    xout, yout, zout : ndarrays
        Arrays of shape (nsubvol, ) storing the points
        belonging to the buffered region defined by subvol_indx.
        Note that the xyz coordinate values may have changed from their original values
        for subvolumes requiring some buffer points to be wrapped around the periodic box.

    cellid : ndarray
        Array of snape (nsubvol, ) storing the cellid of the points in the buffered subvolume.
        Points inside the subvolume are defined by cellid == subvol_indx;
        buffer points are defined by cellid != subvol_indx.

    indx : ndarray
        Integer array of shape (nsubvol, ) storing the indices of the input
        xyz arrays of the points in the buffered subvolume.

        For example, xout == x[indx], except for points that have been wrapped.
    """
    period_xyz = _get_3_element_sequence(period)
    rmax_xyz = _get_3_element_sequence(rmax)

    x, y, z, ix, iy, iz, all_cellids = rectangular_subvolume_cellnum(
        x, y, z, nx, ny, nz, period)

    gen = subvol_bounds_generator(rank, nranks, nx, ny, nz, period_xyz)
    for subvol_bounds in gen:
        subvol_indx, xyz_mins, xyz_maxs = subvol_bounds
        xout, yout, zout, indx, inside_subvol = points_in_buffered_rectangle(
            x, y, z, xyz_mins, xyz_maxs, rmax_xyz, period_xyz)
        cellid = all_cellids[indx]
        yield subvol_indx, xout, yout, zout, cellid, indx


def _get_3_element_sequence(s):
    s_xyz = np.atleast_1d(s)
    if s_xyz.size == 1:
        s_xyz = np.array((s_xyz[0], s_xyz[0], s_xyz[0]))
    elif s_xyz.size != 3:
        raise ValueError("quantity must be a float or 3-element sequence")
    return s_xyz
