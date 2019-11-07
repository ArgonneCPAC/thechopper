"""
"""
import numpy as np
from scipy.spatial import cKDTree
from ..buffered_subvolume_calculations import points_in_buffered_rectangle
from ..buffered_subvolume_calculations import calculate_subvolume_id


def generate_3d_regular_mesh(npts_per_dim, dmin, dmax):
    """
    Function returns a regular 3d grid of npts_per_dim**3 points.

    The spacing of the grid is defined by delta = (dmax-dmin)/npts_per_dim.
    In each dimension, the first point has coordinate delta/2.,
    and the last point has coordinate dmax - delta/2.

    For example, generate_3d_regular_mesh(5, 0, 1) will occupy the 3d grid spanned by
    {0.1, 0.3, 0.5, 0.7, 0.9}.

    Parameters
    -----------
    npts_per_dim : int
        Number of desired points per dimension.

    dmin, dmax : float
        Min/max coordinate value of the box enclosing the grid.

    Returns
    ---------
    x, y, z : array_like
        Three ndarrays of length npts_per_dim**3


    """
    x = np.linspace(dmin, dmax, npts_per_dim+1)
    y = np.linspace(dmin, dmax, npts_per_dim+1)
    z = np.linspace(dmin, dmax, npts_per_dim+1)
    delta = np.diff(x)[0]/2.
    x, y, z = np.array(np.meshgrid(x[:-1], y[:-1], z[:-1]))
    return x.flatten()+delta, y.flatten()+delta, z.flatten()+delta


def test1():
    """Enforce that parallel pair counts with thechopper agree exactly
    with serial pair counts for a set of randomly distributed points.
    """
    rng = np.random.RandomState(43)

    logrbins = np.linspace(-1, np.log10(250), 25)
    rbins = 10**logrbins

    subvol_lengths_xyz = np.array((1250, 1250, 1250)).astype('f4')
    rmax_xyz = np.repeat(np.max(rbins), 3).astype('f4')
    period_xyz = np.array((1500, 1500, 1500)).astype('f4')
    xyz_mins = np.array((0, 0, 0)).astype('f4')
    xyz_maxs = xyz_mins + subvol_lengths_xyz

    npts = int(2e4)
    x = rng.uniform(0, period_xyz[0], npts)
    y = rng.uniform(0, period_xyz[1], npts)
    z = rng.uniform(0, period_xyz[2], npts)

    _w = points_in_buffered_rectangle(
        x, y, z, xyz_mins, xyz_maxs, rmax_xyz, period_xyz)
    xout, yout, zout, indx_out, in_subvol_out = _w

    explicit_mask = np.ones(npts).astype(bool)
    explicit_mask &= x >= xyz_mins[0]
    explicit_mask &= y >= xyz_mins[1]
    explicit_mask &= z >= xyz_mins[2]
    explicit_mask &= x < xyz_maxs[0]
    explicit_mask &= y < xyz_maxs[1]
    explicit_mask &= z < xyz_maxs[2]

    sample1 = [x[explicit_mask], y[explicit_mask], z[explicit_mask]]
    sample1_tree = cKDTree(np.vstack(sample1).T, boxsize=period_xyz)

    sample2 = [x, y, z]
    sample2_tree = cKDTree(np.vstack(sample2).T, boxsize=period_xyz)

    counts_scipy = sample1_tree.count_neighbors(sample2_tree, rbins)

    sample3 = [xout[in_subvol_out], yout[in_subvol_out], zout[in_subvol_out]]
    sample3_tree = cKDTree(np.vstack(sample3).T)

    sample4 = [xout, yout, zout]
    sample4_tree = cKDTree(np.vstack(sample4).T)

    counts_aph = sample3_tree.count_neighbors(sample4_tree, rbins)

    assert np.allclose(counts_scipy, counts_aph, rtol=0.0001), "Wrong pair counts!"


def test2():
    """Require that the calculate_subvolume_id function
    returns a cellnum array with all points lying within [0, nx*ny*nz)
    """
    rng = np.random.RandomState(43)

    npts = int(1e2)
    period = [200, 300, 800]
    x = rng.uniform(0, period[0], npts)
    y = rng.uniform(0, period[1], npts)
    z = rng.uniform(0, period[2], npts)
    nx, ny, nz = 5, 6, 7
    _result = calculate_subvolume_id(x, y, z, nx, ny, nz, period)
    x2, y2, z2, ix, iy, iz, cellnum = _result
    assert np.all(cellnum >= 0)
    assert np.all(cellnum < nx*ny*nz)
    assert np.all(x == x2)
    assert np.all(y == y2)
    assert np.all(z == z2)


def test3():
    """Require that calculate_subvolume_id function wraps xyz points
    lying outside the box back into the box.
    """
    Lbox = 1.
    npts_per_dim = 5
    x, y, z = generate_3d_regular_mesh(npts_per_dim, 0, Lbox)
    x[0] = -0.5
    y[0] = -0.5
    z[0] = -0.5

    nx, ny, nz = npts_per_dim, npts_per_dim, npts_per_dim
    _result = calculate_subvolume_id(x, y, z, nx, ny, nz, Lbox)
    x2, y2, z2, ix, iy, iz, cellnum = _result
    assert np.all(cellnum >= 0)
    assert np.all(cellnum < nx*ny*nz)
    assert np.all(x2 >= 0)
    assert np.all(y2 >= 0)
    assert np.all(z2 >= 0)
    assert np.any(x < 0)
    assert np.any(y < 0)
    assert np.any(z < 0)


def test4():
    """Place a single point at the center of each subvolume and ensure that
    the cellnum array returned by calculate_subvolume_id is correct.
    """
    Lbox = 1.
    npts_per_dim = 5
    x, y, z = generate_3d_regular_mesh(npts_per_dim, 0, Lbox)

    nx, ny, nz = npts_per_dim, npts_per_dim, npts_per_dim
    _result = calculate_subvolume_id(x, y, z, nx, ny, nz, Lbox)
    x2, y2, z2, ix, iy, iz, cellnum = _result

    #  Every cell gets exactly one point
    for icell in range(nx*ny*nz):
        mask = cellnum == icell
        assert np.count_nonzero(mask) == 1

    #  Check a few specific cases
    mask_cellnum0 = cellnum == 0
    assert np.all(x[mask_cellnum0] == np.array((0.1, )))
    assert np.all(y[mask_cellnum0] == np.array((0.1, )))
    assert np.all(z[mask_cellnum0] == np.array((0.1, )))
