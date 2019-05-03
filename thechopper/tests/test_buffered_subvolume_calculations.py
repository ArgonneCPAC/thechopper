"""
"""
import numpy as np
from scipy.spatial import cKDTree
from ..buffered_subvolume_calculations import points_in_buffered_rectangle
from ..buffered_subvolume_calculations import rectangular_subvolume_cellnum


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
    npts = 100
    period = [200, 300, 800]
    x = np.random.uniform(0, period[0], npts)
    y = np.random.uniform(0, period[1], npts)
    z = np.random.uniform(0, period[2], npts)
    nx, ny, nz = 5, 6, 7
    _result = rectangular_subvolume_cellnum(x, y, z, nx, ny, nz, period)
    x2, y2, z2, ix2, iy2, iz2, cellnum2 = _result


