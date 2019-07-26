"""
Script loops over a collection of models, computes summary statistics in parallel
based on thechopper, and writes the results to disk. Computations are done on dummy halo data.

To execute the script:

mpiexec -n 2 python run_dummy_agh_param_scan.py chopper_results.txt 2000 1000 15 5 5 4 2 30

For information on the arguments:

$ python run_dummy_agh_param_scan.py -h

The strategy is to divide the full set of models into `nbatches`,
where each batch contains `nwalkers` points in parameter space.
For each parameter batch, the MPI rank (re)loads its halo data,
and computes the summary statistic contribution for each model in the batch
and for each subvolume belonging to the rank.
The ranks then exchange summary statistics for each model,
write the results to disk, and move on to the next batch of parameters.
"""
import argparse
from mpi4py import MPI
import numpy as np
from chopped_data_generator import generate_chopped_data
from scipy.special import erf
from scipy.spatial import cKDTree
from io_helpers import _get_formatted_output_line, _get_first_header_line


def get_param_batch(nwalkers, batch_num):
    """Return a list of length nwalkers, where each entry is a dictionary storing
    a new collection of randomly selected model parameters.
    """
    rng_params = np.random.RandomState(batch_num)
    logMmin_batch = rng_params.uniform(11.5, 13, nwalkers)
    sigma_logM_batch = rng_params.uniform(0.01, 1, nwalkers)
    list_of_param_dictionaries = list(
        dict(logMmin=a, sigma_logM=b) for a, b in zip(logMmin_batch, sigma_logM_batch))
    return list_of_param_dictionaries


def _get_model_param_names():
    """Grab a dummy batch of parameters and parse it to return the model parameter names
    """
    _nwalkers, _batch_num = 2, 0
    return list(get_param_batch(_nwalkers, _batch_num)[0].keys())


def calculate_subvol_summary_stat_contributions(params, halo_data, **metadata):
    """Given the model parameters and halo data, compute the total number of galaxies,
    and count pairs of galaxies according to the input rbins.
    Return the results after bundling into a dictionary.
    """
    #  Compute weighted number counts
    logMhalo = np.log10(halo_data['mass'])
    erf_arg = (logMhalo - params['logMmin'])/params['sigma_logM']
    mean_ncen = 0.5*(1. + erf(erf_arg))
    ngal_tot = np.sum(mean_ncen)

    #  Calculate pair counts
    x1 = halo_data['x'][~halo_data['buffer_halo']]
    y1 = halo_data['y'][~halo_data['buffer_halo']]
    z1 = halo_data['z'][~halo_data['buffer_halo']]
    pos1 = np.vstack((x1, y1, z1)).T
    tree1 = cKDTree(pos1, boxsize=None)

    x2 = halo_data['x']
    y2 = halo_data['y']
    z2 = halo_data['z']
    pos2 = np.vstack((x2, y2, z2)).T
    tree2 = cKDTree(pos2, boxsize=None)

    pair_counts = tree1.count_neighbors(tree2, metadata['rbins'])

    predictions = dict(nd=ngal_tot, tpcf=pair_counts)
    return predictions


if __name__ == "__main__":
    """
    """
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("foutname", type=str, help="Name of the output data file")
    parser.add_argument("npts", type=int, help="Number of dummy data points")
    parser.add_argument("period", type=float, help="Length of the periodic box")
    parser.add_argument("nwalkers", type=int, help="Number of models to evaluate per box")
    parser.add_argument("nbatches", type=int, help="Number of batches of models to evaluate")
    parser.add_argument("nx", type=int, help="Number of box divisions in the x-dimension")
    parser.add_argument("ny", type=int, help="Number of box divisions in the y-dimension")
    parser.add_argument("nz", type=int, help="Number of box divisions in the z-dimension")
    parser.add_argument("rmax", type=float, help="Maximum search length")
    parser.add_argument("-seed", type=int, help="Random number seed", default=43)
    args = parser.parse_args()
    foutname, npts, period, nwalkers, nbatches, nx, ny, nz, rmax, seed = (
        args.foutname, args.npts, args.period, args.nwalkers, args.nbatches,
        args.nx, args.ny, args.nz, args.rmax, args.seed)
    if rank == 0:
        print("Running {0} model iterations with {1} walkers".format(nbatches, nwalkers))

    num_rbins = 5
    rbins = np.logspace(-0.5, np.log10(rmax), num_rbins)
    simulation_metadata = dict(period=period, rbins=rbins)

    # Generate a full box of random halo data
    rng = np.random.RandomState(seed)
    xyz = rng.uniform(0, 1, 3*npts)*period
    x = xyz[:npts]
    y = xyz[npts:2*npts]
    z = xyz[2*npts:]
    mass = 10**(6*(1. - rng.power(2, size=npts)) + 10)

    with open(foutname, 'w') as fout:

        #  Initialize the accumulators and write the header
        _results = dict(nd=np.zeros(1), tpcf=np.zeros(num_rbins))
        _model_params = np.zeros(2)
        header = _get_first_header_line(_results, _model_params)
        if rank == 0:
            fout.write(header)

        #  Outermost loop over the number of batches of parameters
        for ibatch in range(nbatches):
            param_batch = get_param_batch(nwalkers, ibatch)

            #  We will acculumate our computation into batch_results, a dictionary of dictionaries
            #  Each key of batch_results is an integer corresponding to the model number in the batch
            #  Bound to each key is a dictionary storing the accumulated summary statistics
            batch_results = {i: {_s: _results[_s] for _s in _results.keys()} for i in range(len(param_batch))}

            #  Each rank will compute summary statistics for some collection of subvolumes
            #  Here we loop over each subvolume belonging to the rank
            gen = generate_chopped_data(x, y, z, rank, nranks, nx, ny, nz, period, rmax)
            for subvol_data in gen:
                subvol_indx, xout, yout, zout, cellid, indx = subvol_data
                _halo_data = dict(x=xout, y=yout, z=zout,
                        mass=mass[indx], buffer_halo=cellid!=subvol_indx)

                #  For each subvolume processed by the rank,
                #  loop over all the models in the batch
                #  and accumulate the contributions to the summary statistics
                for imodel, model_params in enumerate(param_batch):
                    summary_stat_contributions = calculate_subvol_summary_stat_contributions(
                        model_params, _halo_data, **simulation_metadata)
                    for sumstat_name, sumstat_value in summary_stat_contributions.items():
                        running_value = batch_results[imodel][sumstat_name]
                        batch_results[imodel][sumstat_name] = running_value + sumstat_value

            #  All subvolumes have now been processed by the rank
            #  Now loop over each model in the batch,
            #  sum the pair counts and histograms computed by all ranks, and write to disk
            for imodel, model_params in enumerate(param_batch):
                for sumstat_name in batch_results[0].keys():
                    sumstat_contribution_from_rank = np.copy(batch_results[imodel][sumstat_name])
                    sumstat = np.empty_like(sumstat_contribution_from_rank)
                    MPI.COMM_WORLD.Allreduce(sumstat_contribution_from_rank, sumstat, op=MPI.SUM)
                    batch_results[imodel][sumstat_name] = sumstat

                param_values = np.array([model_params[pname] for pname in _get_model_param_names()])
                output_line = _get_formatted_output_line(batch_results[imodel], param_values)
                if rank == 0:
                    fout.write(output_line + "\n")
