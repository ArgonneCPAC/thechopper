"""Module tests the correctness of generate_chopped_data through a combination of
brute force pair-count comparisons, and also specific hard-coded cases with easy answers.
"""
import numpy as np
from scipy.spatial import cKDTree
from ..chopped_data_generator import generate_chopped_data, _get_3_element_sequence


def test_correctly_yielded_subvol_indx():
    """Verify that generate_chopped_data yields correct values for the
    subvol_indx variable. This test is a simple hard-coded case that makes it easy
    to explicitly work out the result.
    """
    #  Test config
    nranks = 4
    nx, ny, nz = 2, 2, 2
    rmax, period = 30., 1000.
    npts = int(1e4)
    period_xyz = _get_3_element_sequence(period)
    seed = 43

    x, y, z = _random_3d_rectangular_data(npts, period_xyz, seed=seed)

    #  Rank 0 should get yielded subvolumes 0 and 1
    rank = 0
    gen = generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax)

    #  Check the first yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 0
    __ = _enforce_buffered_bounds(xout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(yout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(zout, 0, period/2., rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], 0, period/2.)
    __ = _enforce_bounds(yout[mask], 0, period/2.)
    __ = _enforce_bounds(zout[mask], 0, period/2.)

    #  Check the second yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 1
    __ = _enforce_buffered_bounds(xout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(yout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(zout, period/2., period, rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], 0, period/2.)
    __ = _enforce_bounds(yout[mask], 0, period/2.)
    __ = _enforce_bounds(zout[mask], period/2., period)

    #  Rank 0 should get no more subvolumes
    try:
        subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    except StopIteration:
        pass
    else:
        raise ValueError("Generated too many subvolumes")

    #  Rank 1 should get yielded subvolumes 2 and 3
    rank = 1
    gen = generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax)

    #  Check the first yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 2
    __ = _enforce_buffered_bounds(xout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(yout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(zout, 0, period/2., rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], 0, period/2.)
    __ = _enforce_bounds(yout[mask], period/2., period)
    __ = _enforce_bounds(zout[mask], 0, period/2.)

    #  Check the second yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 3
    __ = _enforce_buffered_bounds(xout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(yout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(zout, period/2., period, rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], 0, period/2.)
    __ = _enforce_bounds(yout[mask], period/2., period)
    __ = _enforce_bounds(zout[mask], period/2., period)

    #  Rank 1 should get no more subvolumes
    try:
        subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    except StopIteration:
        pass
    else:
        raise ValueError("Generated too many subvolumes")

    #  Rank 2 should get yielded subvolumes 4 and 5
    rank = 2
    gen = generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax)

    #  Check the first yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 4
    __ = _enforce_buffered_bounds(xout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(yout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(zout, 0, period/2., rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], period/2., period)
    __ = _enforce_bounds(yout[mask], 0, period/2.)
    __ = _enforce_bounds(zout[mask], 0, period/2.)

    #  Check the second yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 5
    __ = _enforce_buffered_bounds(xout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(yout, 0, period/2., rmax)
    __ = _enforce_buffered_bounds(zout, period/2., period, rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], period/2., period)
    __ = _enforce_bounds(yout[mask], 0, period/2.)
    __ = _enforce_bounds(zout[mask], period/2., period)

    #  Rank 2 should get no more subvolumes
    try:
        subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    except StopIteration:
        pass
    else:
        raise ValueError("Generated too many subvolumes")

    #  Rank 3 should get yielded subvolumes 6 and 7
    rank = 3
    gen = generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax)

    #  Check the first yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 6
    __ = _enforce_buffered_bounds(xout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(yout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(zout, 0, period/2., rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], period/2., period)
    __ = _enforce_bounds(yout[mask], period/2., period)
    __ = _enforce_bounds(zout[mask], 0, period/2.)

    #  Check the second yielded subvolume
    subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    assert subvol_indx == 7
    __ = _enforce_buffered_bounds(xout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(yout, period/2., period, rmax)
    __ = _enforce_buffered_bounds(zout, period/2., period, rmax)

    mask = cellid == subvol_indx
    __ = _enforce_bounds(xout[mask], period/2., period)
    __ = _enforce_bounds(yout[mask], period/2., period)
    __ = _enforce_bounds(zout[mask], period/2., period)

    #  Rank 3 should get no more subvolumes
    try:
        subvol_indx, xout, yout, zout, cellid, indx = next(gen)
    except StopIteration:
        pass
    else:
        raise ValueError("Generated too many subvolumes")


def test_chopped_pair_counts():
    """Verify that generate_chopped_data can be used to correctly count pairs in
    parallel by enforcing exact agreement with the serial counts.
    """
    #  Test config
    nranks = 4
    nx, ny, nz = 2, 2, 2
    rmax, period = 30., 1000.
    npts = int(1e4)
    period_xyz = _get_3_element_sequence(period)
    seed = 43
    rmin = 0.5
    nbins = 10
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins)

    x, y, z = _random_3d_rectangular_data(npts, period_xyz, seed=seed)
    pos = np.vstack((x, y, z)).T
    tree = cKDTree(pos, boxsize=period)

    serial_pair_counts = tree.count_neighbors(tree, rbins)

    parallel_pair_counts = np.zeros_like(serial_pair_counts)
    for rank in range(nranks):
        gen = generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax)
        for subvol_data in gen:
            subvol_indx, xout, yout, zout, cellid, indx = subvol_data
            mask = cellid == subvol_indx
            tree1 = cKDTree(np.vstack((xout[mask], yout[mask], zout[mask])).T)
            tree2 = cKDTree(np.vstack((xout, yout, zout)).T)
            subvol_counts = tree1.count_neighbors(tree2, rbins)
            parallel_pair_counts = parallel_pair_counts + subvol_counts

    assert np.all(parallel_pair_counts == serial_pair_counts)


def _enforce_buffered_bounds(s, smin, smax, sbuffer):
    """
    """
    assert np.all(s > smin - sbuffer)
    assert np.all(s < smax + sbuffer)

    assert not np.all(s > smin)
    assert not np.all(s < smax)


def _enforce_bounds(s, smin, smax):
    """
    """
    assert np.all(s > smin)
    assert np.all(s < smax)


def _random_3d_rectangular_data(npts, period, seed=43):
    """
    """
    period_xyz = _get_3_element_sequence(period)

    rng = np.random.RandomState(seed)
    xyz = rng.uniform(0, 1, 3*npts)
    x = xyz[:npts]*period_xyz[0]
    y = xyz[npts:2*npts]*period_xyz[1]
    z = xyz[2*npts:]*period_xyz[2]
    return x, y, z
