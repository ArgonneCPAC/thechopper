"""Functions used to decompose the domain of a three-dimensional rectangular volume
with periodic boundary conditions.
"""
import numpy as np


__all__ = ('rectangular_subvolume_cellnum', 'points_in_buffered_rectangle',
        'points_in_rectangle', 'subvol_bounds_generator')


def rectangular_subvolume_cellnum(x, y, z, nx, ny, nz, period):
    """After wrapping the input x, y, z into the periodic box, assign each point
    to a rectangular subvolume defined by the number of subdivisions in each dimension.

    Parameters
    ----------
    x, y, z : ndarrays of shape (npts, )

    nx, ny, nz : integers

    period : float or 3-element sequence

    Returns
    -------
    x, y, z : ndarrays of shape (npts, )
        Identical to the input points except for cases where points have been
        wrapped around the periodic boundaries

    ix, iy, iz : ndarrays of shape (npts, )
        Integer arrays that together specify the subvolume of each point

    cellnum : ndarray of shape (npts, )
        Integer array that by itself specifies the subvolume of each point,
        e.g., (ix, iy, iz)=(0, 0, 0) <==> cellnum=0
    """
    period = np.atleast_1d(period)
    if period.shape[0] == 1:
        period = np.zeros(3) + period[0]
    elif period.shape[0] != 3:
        raise ValueError("period should be float or 3-element sequence")

    x = np.atleast_1d(np.mod(x, period[0]))
    y = np.atleast_1d(np.mod(y, period[1]))
    z = np.atleast_1d(np.mod(z, period[2]))

    _rescaled_x = nx*x/period[0]
    _rescaled_y = ny*y/period[1]
    _rescaled_z = nz*z/period[2]

    ix = np.floor(_rescaled_x).astype('i4')
    iy = np.floor(_rescaled_y).astype('i4')
    iz = np.floor(_rescaled_z).astype('i4')

    cellnum = np.ravel_multi_index( (ix, iy, iz), (nx, ny, nz) )
    return x, y, z, ix, iy, iz, cellnum


def points_in_buffered_rectangle(x, y, z, xyz_mins, xyz_maxs, rmax_xyz, period_xyz):
    """Return the subset of points inside a rectangular subvolume
    surrounded by a buffer region, accounting for periodic boundary conditions.

    Parameters
    ----------
    x, y, z : ndarrays, each with shape (npts, )

    xyz_mins : 3-element sequence
        xyz coordinates of the lower corner of the rectangular subvolume.
        Must have 0 <= xyz_mins <= xyz_maxs <= period_xyz

    xyz_maxs : 3-element sequence
        xyz coordinates of the upper corner of the rectangular subvolume.
        Must have 0 <= xyz_mins <= xyz_maxs <= period_xyz

    rmax_xyz : 3-element sequence
        Search radius distance in the xyz direction.
        Must have rmax_xyz <= period_xyz/2.

    period_xyz : 3-element sequence
        Length of the periodic box

    Returns
    -------
    xout, yout, zout : ndarrays, each with shape (npts_buffered_subvol, )
        Coordinates of points that lie within the search radius of the rectangular subvolume.

        The returned points will lie in the range [xyz_mins-rmax_xyz, xyz_maxs+rmax_xyz],
        which may spill beyond the range [0, Lbox], as required by the size of
        the search radius, the size of the box, and the position of the subvolume.

        The buffered subvolume includes all points relevant to pair-counting
        within rmax_xyz for points in the rectangular subvolume,
        and so periodic boundary conditions can be ignored for xout, yout, zout.

    indx : ndarray, shape (npts_buffered_subvol, )
        Index of the corresponding point in the input xyz arrays.

        xout[i] == x[indx[i]] except for cases where the point
        has been wrapped around the periodic boundaries.

    inside_subvol : ndarray, shape (npts_buffered_subvol, )
        boolean array is True when the point is in the rectangular subvolume,
        False when the point is in the +/-rmax_xyz region surrounding the rectangular subvolume.
    """
    xyz_mins = np.array(xyz_mins)
    xyz_maxs = np.array(xyz_maxs)
    rmax_xyz = np.array(rmax_xyz)
    period_xyz = np.array(period_xyz)
    x = np.mod(x, period_xyz[0])
    y = np.mod(y, period_xyz[1])
    z = np.mod(z, period_xyz[2])

    x_collector = []
    y_collector = []
    z_collector = []
    indx_collector = []
    in_subvol_collector = []

    for subregion in _buffering_rectangular_subregions(xyz_mins, xyz_maxs, rmax_xyz):
        subregion_ix_iy_iz, subregion_xyz_mins, subregion_xyz_maxs = subregion

        subregion_x, subregion_y, subregion_z, subregion_indx = points_in_rectangle(
            x, y, z, subregion_xyz_mins, subregion_xyz_maxs, period_xyz)

        _npts = len(subregion_x)
        if _npts > 0:
            x_collector.append(subregion_x)
            y_collector.append(subregion_y)
            z_collector.append(subregion_z)
            indx_collector.append(subregion_indx)

            in_subvol = np.zeros_like(subregion_x).astype(bool) + (subregion_ix_iy_iz == (0, 0, 0))
            in_subvol_collector.append(in_subvol)

    if len(x_collector) == 0:
        xout = np.zeros(0, dtype='f8')
        yout = np.zeros(0, dtype='f8')
        zout = np.zeros(0, dtype='f8')
        indx = np.zeros(0, dtype='i8')
        inside_subvol = np.zeros(0, dtype=bool)
    else:
        xout = np.concatenate(x_collector).astype(float)
        yout = np.concatenate(y_collector).astype(float)
        zout = np.concatenate(z_collector).astype(float)
        indx = np.concatenate(indx_collector).astype(int)
        inside_subvol = np.concatenate(in_subvol_collector).astype(bool)

    return xout, yout, zout, indx, inside_subvol


def points_in_rectangle(x, y, z, xyz_mins, xyz_maxs, period_xyz):
    """Return the set of all points located in the rectangular subvolume [xyz_mins, xyz_maxs).

    Periodic boundary conditions are accounted for by including any points that
    fall inside the subvolume when one or more the coordinates is shifted by +/- period.

    Parameters
    ----------
    x, y, z : ndarrays, each with shape (npts, )

    xyz_mins : 3-element sequence
        xyz coordinates of the lower corner of the rectangular subvolume.
        Must have 0 <= xyz_mins <= xyz_maxs <= period_xyz

    xyz_maxs : 3-element sequence
        xyz coordinates of the upper corner of the rectangular subvolume.
        Must have 0 <= xyz_mins <= xyz_maxs <= period_xyz

    period_xyz : 3-element sequence
        Length of the periodic box

    Returns
    -------
    xout, yout, zout : ndarrays, each with shape (npts_subvol, )
        Coordinates of points that lie within the rectangular subvolume.
        All points will lie in the range [xyz_mins, xyz_maxs]

    indx : ndarray, shape (npts_subvol, )
        index of the corresponding point in the input xyz arrays
    """
    xyz_mins = np.array(xyz_mins)
    xyz_maxs = np.array(xyz_maxs)
    period_xyz = np.array(period_xyz)

    x_collector = []
    y_collector = []
    z_collector = []
    indx_collector = []

    npts_input_data = len(x)
    available_indices = np.arange(npts_input_data).astype(int)

    for mask, shift in _pbc_generator_mask_and_shift(x, y, z, xyz_mins, xyz_maxs, period_xyz):

        _npts = np.count_nonzero(mask)
        if _npts > 0:
            x_collector.append(x[mask] - shift[0]*period_xyz[0])
            y_collector.append(y[mask] - shift[1]*period_xyz[1])
            z_collector.append(z[mask] - shift[2]*period_xyz[2])
            indx_collector.append(available_indices[mask])

    if len(x_collector) == 0:
        xout = np.zeros(0, dtype='f8')
        yout = np.zeros(0, dtype='f8')
        zout = np.zeros(0, dtype='f8')
        indx = np.zeros(0, dtype='i8')
    else:
        xout = np.concatenate(x_collector).astype(float)
        yout = np.concatenate(y_collector).astype(float)
        zout = np.concatenate(z_collector).astype(float)
        indx = np.concatenate(indx_collector).astype(int)
    return xout, yout, zout, indx


def subvol_bounds_generator(rank, nranks, num_xdivs, num_ydivs, num_zdivs, period):
    """
    """
    num_tot_subvols = num_xdivs*num_ydivs*num_zdivs
    subvol_indices_all = np.arange(num_tot_subvols)
    subvol_indices_rank = np.array_split(subvol_indices_all, nranks)[rank]

    for subvol_indx in subvol_indices_rank:
        ix, iy, iz = np.unravel_index(
            subvol_indx, (num_xdivs, num_ydivs, num_zdivs) )
        xlo, xhi = _get_subvol_bounds_1d(ix, period[0], num_xdivs)
        ylo, yhi = _get_subvol_bounds_1d(iy, period[1], num_ydivs)
        zlo, zhi = _get_subvol_bounds_1d(iz, period[2], num_zdivs)
        xyz_mins = (xlo, ylo, zlo)
        xyz_maxs = (xhi, yhi, zhi)
        yield subvol_indx, xyz_mins, xyz_maxs


def _get_subvol_bounds_1d(ip, dim_length, dim_ndivs):
    ds = dim_length/float(dim_ndivs)
    return ip*ds, (ip+1)*ds


def _pbc_generator_mask_and_shift(x, y, z, xyz_mins, xyz_maxs, period_xyz):
    """Generate masks for 27 rectangular subvolumes obtained by
    by shifting the input rectangular subvolume by +/0/- period in each dimension
    """
    for bounds in _pbc_generator_xyz_bounds(xyz_mins, xyz_maxs, period_xyz):
        subvol_xyz_shift, subvol_xyz_mins, subvol_xyz_maxs = bounds

        mask = np.ones_like(x).astype(bool)
        mask &= x >= subvol_xyz_mins[0]
        mask &= y >= subvol_xyz_mins[1]
        mask &= z >= subvol_xyz_mins[2]
        mask &= x < subvol_xyz_maxs[0]
        mask &= y < subvol_xyz_maxs[1]
        mask &= z < subvol_xyz_maxs[2]

        yield mask, subvol_xyz_shift


def _pbc_generator_xyz_bounds(xyz_mins, xyz_maxs, period_xyz):
    """Generate the bounds for 27 rectangular subvolumes obtained by
    by shifting the input rectangular subvolume by +/0/- period in each dimension
    """
    for ix in (-1, 0, 1):
        xmin = xyz_mins[0] + ix*period_xyz[0]
        xmax = xyz_maxs[0] + ix*period_xyz[0]

        for iy in (-1, 0, 1):
            ymin = xyz_mins[1] + iy*period_xyz[1]
            ymax = xyz_maxs[1] + iy*period_xyz[1]

            for iz in (-1, 0, 1):
                zmin = xyz_mins[2] + iz*period_xyz[2]
                zmax = xyz_maxs[2] + iz*period_xyz[2]

                yield (ix, iy, iz), (xmin, ymin, zmin), (xmax, ymax, zmax)


def _buffering_rectangular_subregions(xyz_mins, xyz_maxs, rmax_xyz):
    """Decompose the buffered subvolume into 27 subregions:
    one subregion for the subvolume itself, and the remaining 26 come from
    the rectanguloids adjacent to each face, edge, and corner.

    In particular, our buffered subvolume is decomposed into the following 27 subregions:

    * 1 for the data in the rectangular subvolume specified by (xyz_mins, xyz_maxs);
    * 6 for the buffer data in the region rmax_xyz beyond each face of the rectangular subvolume;
    * 12 for the buffer data in the region rmax_xyz beyond each edge of the rectangular subvolume;
    * 8 for the buffer data in the region rmax_xyz beyond each corner of the rectangular subvolume;

    The _buffering_rectangular_subregions generator loops over these 27 subvolumes,
    and yields their xyz boundaries.
    """
    for ix in (-1, 0, 1):
        xmin, xmax = _get_buffering_subregion_minmax(ix, xyz_mins[0], xyz_maxs[0], rmax_xyz[0])

        for iy in (-1, 0, 1):
            ymin, ymax = _get_buffering_subregion_minmax(iy, xyz_mins[1], xyz_maxs[1], rmax_xyz[1])

            for iz in (-1, 0, 1):
                zmin, zmax = _get_buffering_subregion_minmax(iz, xyz_mins[2], xyz_maxs[2], rmax_xyz[2])

                yield (ix, iy, iz), (xmin, ymin, zmin), (xmax, ymax, zmax)


def _get_buffering_subregion_minmax(ip, pmin, pmax, rmax):
    """Given a line segment (pmin, pmax) buffered by a region rmax,
    calculate the segment (smin, smax) that either buffers or duplicates (pmin, pmax),
    depending on the input variable ip.
    """
    if ip == -1:
        smin, smax = pmin - rmax, pmin
    elif ip == 0:
        smin, smax = pmin, pmax
    elif ip == 1:
        smin, smax = pmax, pmax + rmax
    return smin, smax
