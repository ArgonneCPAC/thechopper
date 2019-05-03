"""
"""
import numpy as np
from scipy.spatial import cKDTree
from ..buffered_subvolume_calculations import points_in_buffered_rectangle
from ..buffered_subvolume_calculations import rectangular_subvolume_cellnum


def generate_3d_regular_mesh(npts_per_dim, dmin, dmax):
    """
    Function returns a regular 3d grid of npts_per_dim**3 points.

    The spacing of the grid is defined by delta = (dmax-dmin)/npts_per_dim.
    In each dimension, the first point has coordinate delta/2.,
    and the last point has coordinate dmax - delta/2.

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
    # return np.vstack([x.flatten()+delta, y.flatten()+delta, z.flatten()+delta]).T


def test1():

    logrbins = np.linspace(-1, np.log10(250), 25)
    rbins = 10**logrbins

    subvol_lengths_xyz = np.array((1250, 1250, 1250)).astype(float)
    rmax_xyz = np.repeat(np.max(rbins), 3).astype(float)
    period_xyz = np.array((1500, 1500, 1500)).astype(float)
    xyz_mins = np.array((0, 0, 0)).astype(float)
    xyz_maxs = xyz_mins + subvol_lengths_xyz

    npts = int(2e4)
    xyz = np.random.random((npts, 3))*period_xyz
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    _w = points_in_buffered_rectangle(x, y, z, xyz_mins, xyz_maxs, rmax_xyz, period_xyz)
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

    assert np.all(counts_scipy == counts_aph), "Wrong pair counts!"


def test2():
    """
    """
    npts = int(1e2)
    period = [200, 300, 800]
    x = np.random.uniform(0, period[0], npts)
    y = np.random.uniform(0, period[1], npts)
    z = np.random.uniform(0, period[2], npts)
    nx, ny, nz = 5, 6, 7
    _result = rectangular_subvolume_cellnum(x, y, z, nx, ny, nz, period)
    x2, y2, z2, ix2, iy2, iz2, cellnum2 = _result
    assert np.all(cellnum2 >= 0)
    assert np.all(cellnum2 < nx*ny*nz)
    assert np.all(x == x2)
    assert np.all(y == y2)
    assert np.all(z == z2)


def test3():
    """
    """
    npts_per_dim = 20

    x, y, z = generate_3d_regular_mesh(npts_per_dim, -500, 2500)

    nx, ny, nz = 3, 4, 5
    _result = rectangular_subvolume_cellnum(x, y, z, nx, ny, nz, 1500)
    x2, y2, z2, ix2, iy2, iz2, cellnum2 = _result
    assert np.all(cellnum2 >= 0)
    assert np.all(cellnum2 < nx*ny*nz)
    assert np.any(x != x2)
    assert np.any(y != y2)
    assert np.any(z != z2)

    mask = cellnum2 == 0
    assert np.all(x2[mask] < 1500/float(nx))
    assert np.all(y2[mask] < 1500/float(ny))
    assert np.all(z2[mask] < 1500/float(nz))

    mask = cellnum2 == nx*ny*nz - 1
    assert np.all(x2[mask] >= 1500-1500/float(nx))
    assert np.all(y2[mask] >= 1500-1500/float(ny))
    assert np.all(z2[mask] >= 1500-1500/float(nz))

