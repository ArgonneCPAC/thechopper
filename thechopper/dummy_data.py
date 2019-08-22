"""
"""
import numpy as np
import string


DEFAULT_SEED = 43


def generate_random_string(string_length=16, seed=DEFAULT_SEED):
    anum = string.ascii_letters + string.digits
    rng = np.random.RandomState(seed)
    return ''.join(rng.choice([s for s in anum], string_length, replace=True))


def dummy_halo_properties(npts, nranks, Lbox, seed=DEFAULT_SEED):
    """
    """
    core_id = np.arange(npts).astype('i8')
    x = np.random.RandomState(seed).uniform(0, Lbox, npts).astype('f4')
    y = np.random.RandomState(seed+1).uniform(0, Lbox, npts).astype('f4')
    z = np.random.RandomState(seed+2).uniform(0, Lbox, npts).astype('f4')
    mass = 10**np.random.RandomState(seed).uniform(10, 15, npts).astype('f4')
    rank = np.random.RandomState(seed).randint(0, nranks, npts).astype('i4')
    data_origin_fname = generate_random_string(seed)
    data = dict(core_id=core_id, x=x, y=y, z=z, mass=mass, rank=rank)
    dtype_dict = list((('core_id', 'i8'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('mass', 'f4'), ('rank', 'i4')))
    dtype_dict = dict(core_id='i8', x='f4', y='f4', z='f4', mass='f4', rank='i4')
    metadata = dict(data_origin_fname=data_origin_fname, npts=npts, dtype_dict=dtype_dict)
    return data, metadata
