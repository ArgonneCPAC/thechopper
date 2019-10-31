"""
"""
import numpy as np
import string


DEFAULT_SEED = 43


def generate_random_string(string_length=16, seed=DEFAULT_SEED):
    anum = string.ascii_letters + string.digits
    rng = np.random.RandomState(seed)
    return ''.join(rng.choice([s for s in anum], string_length, replace=True))


def dummy_halo_properties(npts, nranks, Lbox, seed=DEFAULT_SEED, reader_rank=None,
            box_min=0, core_id_min=0):
    """
    """
    core_id = np.arange(core_id_min, core_id_min+npts).astype('i8')

    box_min = _get_3_element_sequence(box_min)
    period_xyz = _get_3_element_sequence(Lbox)
    xmin, ymin, zmin = box_min
    xmax, ymax, zmax = period_xyz
    x = np.random.RandomState(seed).uniform(xmin, xmax, npts).astype('f4')
    y = np.random.RandomState(seed+1).uniform(ymin, ymax, npts).astype('f4')
    z = np.random.RandomState(seed+2).uniform(zmin, zmax, npts).astype('f4')

    mass = 10**np.random.RandomState(seed).uniform(10, 15, npts).astype('f4')
    rank = np.random.RandomState(seed).randint(0, nranks, npts).astype('i4')
    data_origin_fname = generate_random_string(seed)

    if reader_rank is None:
        mask = np.ones(npts).astype(bool)
    else:
        mask = rank == reader_rank

    data = dict(core_id=core_id[mask],
                x=x[mask], y=y[mask], z=z[mask],
                mass=mass[mask], rank=rank[mask])

    dtype_dict = list((('core_id', 'i8'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('mass', 'f4'), ('rank', 'i4')))
    dtype_dict = dict(core_id='i8', x='f4', y='f4', z='f4', mass='f4', rank='i4')
    metadata = dict(data_origin_fname=data_origin_fname, npts=npts, dtype_dict=dtype_dict)
    return data, metadata


def concatenate_dummy_data(*dummy_data_collection):
    keys = list(dummy_data_collection[0].keys())
    return {key: np.concatenate([d[key] for d in dummy_data_collection]) for key in keys}


def _get_3_element_sequence(s):
    s_xyz = np.atleast_1d(s)
    if s_xyz.size == 1:
        s_xyz = np.array((s_xyz[0], s_xyz[0], s_xyz[0]))
    elif s_xyz.size != 3:
        raise ValueError("quantity must be a float or 3-element sequence")
    return s_xyz
