"""Functions used to decompose the domain of a three-dimensional rectangular volume
with periodic boundary conditions.
"""
import numpy as np


def get_data_for_rank(comm, data, nx, ny, nz, period, rmax, columns_to_retrieve):
    """Chop the input data and return the subvolume for the input rank."""
    rank, nranks = comm.Get_rank(), comm.Get_size()
    chopped_data = get_chopped_data(
        data, nx, ny, nz, period, rmax, columns_to_retrieve)
    flattened_chopped_data, npts_to_send_to_rank = assign_chopped_data_to_ranks(
        chopped_data, nx, ny, nz, nranks)

    npts_to_receive_from_rank = np.empty_like(npts_to_send_to_rank)
    comm.Alltoall(npts_to_send_to_rank, npts_to_receive_from_rank)
    comm.Barrier()

    data_for_rank = dict()
    deterministic_keylist = sorted(list(flattened_chopped_data.keys()))
    send_counts = npts_to_send_to_rank
    recv_counts = npts_to_receive_from_rank
    for colname in deterministic_keylist:
        sendbuf = flattened_chopped_data[colname]
        recv_buff = np.empty(recv_counts.sum(), dtype=sendbuf.dtype)
        comm.Alltoallv([sendbuf, send_counts], [recv_buff, recv_counts])
        comm.Barrier()
        data_for_rank[colname] = recv_buff
        comm.Barrier()

    cells_assigned_to_rank = _get_cells_assigned_to_ranks(nx, ny, nz, nranks)[rank]
    output_data = dict()
    for cellnum in cells_assigned_to_rank:
        mask = data_for_rank['_subvol_indx'] == cellnum
        output_data[cellnum] = {key: data_for_rank[key][mask] for key in data_for_rank.keys()}

    return output_data


def get_chopped_data(data, nx, ny, nz, period, rmax, columns_to_retrieve):
    """
    Divide the input data into a collection of buffered subvolumes.

    Parameters
    ----------
    data : dict
        Keys are names of galaxy/halo properties. Values are ndarrays.

    nx, ny, nz : integers
        Number of divisions of the periodic box in each dimension

    period : Float or 3-element sequence
        Length of the periodic box in each dimension.
        Box will be assumed to be a cube if passing a float.

    rmax : Float or 3-element sequence
        Search radius distance in each Cartesian direction.
        Must have rmax <= period/2 in each dimension.

    columns_to_retrieve : list of strings

    Returns
    -------
    chopped_data : dict
        Keys are names of galaxy/halo properties.
        The value bound to each key is a list with length equal to the
        total number of subvolumes: nx*ny*nz. Each element of the list stores
        a (possibly empty) ndarray storing data belonging to
        the corresponding buffered subvolume.

    """
    period_xyz = _get_3_element_sequence(period)
    rmax_xyz = _get_3_element_sequence(rmax)

    #  Wrap xyz into the box before assigning data to subvolumes
    data['x'] = data['x'] % period_xyz[0]
    data['y'] = data['y'] % period_xyz[1]
    data['z'] = data['z'] % period_xyz[2]

    #  Assign data to subvolumes
    dx = float(period_xyz[0]/nx)
    dy = float(period_xyz[1]/ny)
    dz = float(period_xyz[2]/nz)
    _ix = np.array(data['x'] // dx).astype(int)
    _iy = np.array(data['y'] // dy).astype(int)
    _iz = np.array(data['z'] // dz).astype(int)
    data['_ix'] = _ix
    data['_iy'] = _iy
    data['_iz'] = _iz

    #  columns_to_retrieve should include _ix, _iy, _iz, _inside_subvol, _subvol_indx
    #  xyz get remapped and so will be treated separately
    _always = {'x', 'y', 'z', '_inside_subvol', '_subvol_indx'}
    _s = set(columns_to_retrieve) - _always
    _cellids = {'_ix', '_iy', '_iz'}
    _t = _s.union(_cellids)
    columns_to_retrieve = list(_t)

    chopped_data = {key: [] for key in columns_to_retrieve}
    chopped_data['x'] = []
    chopped_data['y'] = []
    chopped_data['z'] = []
    chopped_data['_inside_subvol'] = []
    chopped_data['_subvol_indx'] = []

    gen = _subvol_bounds_generator(nx, ny, nz, period_xyz)
    for subvol_bounds in gen:
        subvol_indx, xyz_mins, xyz_maxs = subvol_bounds

        _ret = points_in_buffered_rectangle(data['x'], data['y'], data['z'],
            xyz_mins, xyz_maxs, rmax_xyz, period_xyz)
        xout, yout, zout, indx, inside_subvol = _ret

        chopped_data['x'].append(xout)
        chopped_data['y'].append(yout)
        chopped_data['z'].append(zout)
        chopped_data['_inside_subvol'].append(inside_subvol)

        _subvol_indx = np.zeros(xout.size).astype(int) + subvol_indx
        chopped_data['_subvol_indx'].append(_subvol_indx)

        for colname in columns_to_retrieve:
            chopped_data[colname].append(data[colname][indx])

    return chopped_data


def assign_chopped_data_to_ranks(chopped_data, nx, ny, nz, nranks):
    """Starting from data that has been spatially partitioned into buffered subvolumes
    (i.e., the output of the get_chopped_data function),
    count the number of points assigned to each rank and flatten the chopped data.

    Parameters
    ----------
    chopped_data : dict
        Keys are names of galaxy/halo properties.
        The value bound to each key is a list with length equal to the
        total number of subvolumes: nx*ny*nz. Each element of the list stores
        a (possibly empty) ndarray storing data belonging to
        the corresponding buffered subvolume.

    nx, ny, nz : integers
        Number of divisions of the periodic box in each dimension

    nranks : int
        Number of MPI ranks
    """
    example_key = list(chopped_data.keys())[0]
    npts_in_cell = [arr.size for arr in chopped_data[example_key]]

    cells_assigned_to_ranks = _get_cells_assigned_to_ranks(nx, ny, nz, nranks)

    npts_assigned_to_rank = list(sum(npts_in_cell[i]
            for i in cells_assigned_to_ranks[rank]) for rank in range(nranks))

    flattened_data = {key: np.concatenate(chopped_data[key])
        for key in chopped_data.keys()}

    return flattened_data, np.array(npts_assigned_to_rank).astype(int)


def _get_cells_assigned_to_ranks(nx, ny, nz, nranks):
    ndivs_total = nx*ny*nz
    cells_assigned_to_rank = np.array_split(np.arange(ndivs_total), nranks)
    return cells_assigned_to_rank


def points_in_buffered_rectangle(x, y, z, xyz_mins, xyz_maxs, rmax_xyz, period):
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

    period : Float or 3-element sequence
        Length of the periodic box in each dimension.
        Box will be assumed to be a cube if passing a float.

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
    period_xyz = _get_3_element_sequence(period)
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


def points_in_rectangle(x, y, z, xyz_mins, xyz_maxs, period):
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

    period : Float or 3-element sequence
        Length of the periodic box in each dimension.
        Box will be assumed to be a cube if passing a float.

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
    period_xyz = _get_3_element_sequence(period)

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


def _pbc_generator_mask_and_shift(x, y, z, xyz_mins, xyz_maxs, period):
    """Generate masks for 27 rectangular subvolumes obtained by
    by shifting the input rectangular subvolume by +/0/- period in each dimension
    """
    period_xyz = _get_3_element_sequence(period)

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


def _pbc_generator_xyz_bounds(xyz_mins, xyz_maxs, period):
    """Generate the bounds for 27 rectangular subvolumes obtained by
    by shifting the input rectangular subvolume by +/0/- period in each dimension
    """
    period_xyz = _get_3_element_sequence(period)

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


def _subvol_bounds_generator(num_xdivs, num_ydivs, num_zdivs, period):
    """
    """
    period_xyz = _get_3_element_sequence(period)
    num_tot_subvols = num_xdivs*num_ydivs*num_zdivs

    for subvol_indx in range(num_tot_subvols):
        ix, iy, iz = np.unravel_index(
            subvol_indx, (num_xdivs, num_ydivs, num_zdivs) )
        xlo, xhi = _get_subvol_bounds_1d(ix, period_xyz[0], num_xdivs)
        ylo, yhi = _get_subvol_bounds_1d(iy, period_xyz[1], num_ydivs)
        zlo, zhi = _get_subvol_bounds_1d(iz, period_xyz[2], num_zdivs)
        xyz_mins = (xlo, ylo, zlo)
        xyz_maxs = (xhi, yhi, zhi)
        yield subvol_indx, xyz_mins, xyz_maxs


def _get_subvol_bounds_1d(ip, dim_length, dim_ndivs):
    ds = dim_length/float(dim_ndivs)
    return ip*ds, (ip+1)*ds


def _get_3_element_sequence(s):
    s_xyz = np.atleast_1d(s)
    if s_xyz.size == 1:
        s_xyz = np.array((s_xyz[0], s_xyz[0], s_xyz[0]))
    elif s_xyz.size != 3:
        raise ValueError("quantity must be a float or 3-element sequence")
    return s_xyz
