"""Functions distribute chopped simulation data to MPI ranks."""
import numpy as np
from .buffered_subvolume_calculations import points_in_buffered_rectangle


__all__ = ("get_buffered_subvolumes", "get_all_chopped_data")


def get_buffered_subvolumes(
    comm, catalog, nx, ny, nz, period, rmax, colnames="all", source_rank=0
):
    """Get the buffered subvolume data assigned to the MPI rank.

    Parameters
    ----------
    comm : MPI communicator

    catalog : dict
        For rank == source_rank, keys are column names, values are ndarrays.
        For rank != source_rank, should be an empty dict.

    nx, ny, nz : integers
        Specify how to decompose the simulation into subvolumes

    period : Float or 3-element sequence
        Length of the periodic box in each dimension.
        Box will be assumed to be a cube if period is a float.

    rmax : Float or 3-element sequence
        Search radius in each Cartesian direction.
        Must have rmax <= period/2 in each dimension.

    colnames : string or list of strings, optional
        Specifies which columns to distribute to the ranks.
        Default is all columns.

    source_rank : int, optional
        Rank in possession of the catalog data. Default is 0.

    Returns
    -------
    datalist : list
        List with length equal to the number of subvolumes assigned to the rank.
        Each element is a dictionary storing catalog data in the subvolume.

    ranklist : list
        List with length equal to the number of subvolumes assigned to the rank.
        Each element is an integer providing the subvolume ID.

    """
    rank, nranks = comm.Get_rank(), comm.Get_size()
    if colnames == "all":
        colnames = sorted(list(catalog.keys()))
    else:
        colnames = [s for s in np.atleast_1d(colnames)]

    if rank == source_rank:
        chopped_cat = get_all_chopped_data(catalog, nx, ny, nz, period, rmax, colnames)
    else:
        chopped_cat = dict()

    data_collection = []
    num_subvols = nx * ny * nz
    for isubvol in range(0, num_subvols):
        cat_to_send = dict(
            ((key, chopped_cat[key][isubvol])) for key in chopped_cat.keys()
        )
        dest_rank = _get_rank_responsible_for_subvol_id(isubvol, nx, ny, nz, nranks)
        if dest_rank == 0:
            data_collection.append(cat_to_send)
        else:
            recv_cat = _distribute_simdata(comm, cat_to_send, source_rank, dest_rank)
            data_collection.append(recv_cat)

    datalist = list((d for d in data_collection if len(list(d.keys())) > 0))
    ranklist = list(
        (i for i, d in enumerate(data_collection) if len(list(d.keys())) > 0)
    )
    return datalist, ranklist


def get_all_chopped_data(data, nx, ny, nz, period, rmax, colnames):
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
        Box will be assumed to be a cube if period is a float.

    rmax : Float or 3-element sequence
        Search radius in each Cartesian direction.
        Must have rmax <= period/2 in each dimension.

    colnames : list of strings

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
    data["x"] = data["x"] % period_xyz[0]
    data["y"] = data["y"] % period_xyz[1]
    data["z"] = data["z"] % period_xyz[2]

    #  Assign data to subvolumes
    dx = float(period_xyz[0] / nx)
    dy = float(period_xyz[1] / ny)
    dz = float(period_xyz[2] / nz)
    _ix = np.array(data["x"] // dx).astype("i4")
    _iy = np.array(data["y"] // dy).astype("i4")
    _iz = np.array(data["z"] // dz).astype("i4")
    data["_ix"] = _ix
    data["_iy"] = _iy
    data["_iz"] = _iz

    #  columns_to_retrieve always has the following columns:
    #   _ix, _iy, _iz, _inside_subvol, _subvol_indx
    #  xyz get remapped and so will be treated separately
    _always = {"x", "y", "z", "_inside_subvol", "_subvol_indx"}
    _s = set(colnames) - _always
    _cellids = {"_ix", "_iy", "_iz"}
    _t = _s.union(_cellids)
    columns_to_retrieve = list(_t)

    chopped_data = dict(((key, [])) for key in columns_to_retrieve)
    chopped_data["x"] = []
    chopped_data["y"] = []
    chopped_data["z"] = []
    chopped_data["_inside_subvol"] = []
    chopped_data["_subvol_indx"] = []

    gen = _subvol_bounds_generator(nx, ny, nz, period_xyz)
    for subvol_bounds in gen:
        subvol_indx, xyz_mins, xyz_maxs = subvol_bounds

        _ret = points_in_buffered_rectangle(
            data["x"], data["y"], data["z"], xyz_mins, xyz_maxs, rmax_xyz, period_xyz
        )
        xout, yout, zout, indx, inside_subvol = _ret

        chopped_data["x"].append(xout)
        chopped_data["y"].append(yout)
        chopped_data["z"].append(zout)
        chopped_data["_inside_subvol"].append(inside_subvol)

        _subvol_indx = np.zeros(xout.size).astype("i8") + subvol_indx
        chopped_data["_subvol_indx"].append(_subvol_indx)

        for colname in columns_to_retrieve:
            chopped_data[colname].append(data[colname][indx])

    return chopped_data


def _distribute_simdata(comm, halo_catalog, source, dest, tag=0, columns_to_send="all"):
    """Send data in the halo_catalog from source rank to destination rank.

    Parameters
    ----------
    comm : MPI communicator

    halo_catalog : dict
        Each key is a halo property, each value an ndarray of shape (nhalos, )

    source : int
        Rank of the sender

    dest : int
        Rank of the receiver

    tag : int, optional
        Tag to use in the message passing. Default is 0.

    columns_to_send : string or list of strings, optional
        Default is 'all', for sending all the arrays in halo_catalog

    Returns
    -------
    recv_halocat : dict
        One key for each value in columns_to_send.
        Each value will be an ndarray of shape (nhalos, )

    """
    rank = comm.Get_rank()

    if columns_to_send == "all":
        columns_to_send = sorted(list(halo_catalog.keys()))
    else:
        columns_to_send = [s for s in np.atleast_1d(columns_to_send)]

    if rank == source:
        recv_halocat = dict()

        ncols_to_send = len(columns_to_send)
        comm.send(ncols_to_send, dest=dest, tag=tag)

        for send_colname in columns_to_send:
            send_arr = halo_catalog[send_colname]
            send_npts, send_dtype = send_arr.size, str(send_arr.dtype)
            comm.send(send_colname, dest=dest, tag=tag)
            comm.send(send_npts, dest=dest, tag=tag)
            comm.send(send_dtype, dest=dest, tag=tag)
            comm.Send(send_arr, dest=dest, tag=tag)

    elif rank == dest:
        recv_halocat = dict()

        ncols_to_recv = comm.recv(source=source, tag=tag)

        for __ in range(ncols_to_recv):
            recv_colname = comm.recv(source=source, tag=tag)
            npts_to_recv = comm.recv(source=source, tag=tag)
            dtype_to_recv = comm.recv(source=source, tag=tag)
            recvarray = np.zeros(npts_to_recv).astype(dtype_to_recv)
            comm.Recv(recvarray, source=source, tag=tag)
            recv_halocat[recv_colname] = recvarray

    else:
        recv_halocat = dict()

    return recv_halocat


def _subvol_bounds_generator(nx, ny, nz, period):
    """For every subvolume, yield its ID and its xyz bounds."""
    num_tot_subvols = nx * ny * nz
    subvol_ids = np.arange(num_tot_subvols)

    for subvol_id in subvol_ids:
        ix, iy, iz = np.unravel_index(subvol_id, (nx, ny, nz))
        xlo, xhi = _get_subvol_bounds_1d(ix, period[0], nx)
        ylo, yhi = _get_subvol_bounds_1d(iy, period[1], ny)
        zlo, zhi = _get_subvol_bounds_1d(iz, period[2], nz)
        xyz_mins = (xlo, ylo, zlo)
        xyz_maxs = (xhi, yhi, zhi)
        yield subvol_id, xyz_mins, xyz_maxs


def _get_subvol_bounds_1d(ip, dim_length, dim_ndivs):
    """Get the boundaries of a 1-d line segment."""
    ds = dim_length / float(dim_ndivs)
    return ip * ds, (ip + 1) * ds


def _get_subvol_ids_assigned_to_ranks(nx, ny, nz, nranks):
    """Assign subvolumes to the available ranks."""
    ndivs_total = nx * ny * nz
    subvol_ids_assigned_to_rank = np.array_split(np.arange(ndivs_total), nranks)
    return subvol_ids_assigned_to_rank


def _get_rank_responsible_for_subvol_id(subvol_ID, nx, ny, nz, nranks):
    """Find the reank responsible for the subvolume."""
    cells_assigned_to_ranks = _get_subvol_ids_assigned_to_ranks(nx, ny, nz, nranks)
    _x = [subvol_ID in arr for arr in cells_assigned_to_ranks]
    return _x.index(True)


def _get_3_element_sequence(s):
    """Return a 3-element sequence defined by s.

    If s is already a 3-element sequence, return s unchanged.
    If s is a scalar, return [s, s, s].

    """
    s_xyz = np.atleast_1d(s)
    if s_xyz.size == 1:
        s_xyz = np.array((s_xyz[0], s_xyz[0], s_xyz[0]))
    elif s_xyz.size != 3:
        raise ValueError("quantity must be a float or 3-element sequence")
    return s_xyz
