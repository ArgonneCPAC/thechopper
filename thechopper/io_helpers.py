"""
"""
import numpy as np


def _get_formatted_output_line(results, model_params):
    shapes = [np.size(results[key]) for key in results.keys()]
    ndim_data_vector = sum(shapes)
    data_vector = np.zeros(ndim_data_vector)
    ifirst = 0
    for i, key in enumerate(results.keys()):
        ilast = ifirst + shapes[i]
        data_vector[ifirst:ilast] = results[key]
        ifirst = ilast

    _output = np.array_str(data_vector, suppress_small=True, precision=6)
    output_results = ' '.join(_output.replace('[', '').replace(']', '').split())

    _output = np.array_str(model_params, suppress_small=True, precision=6)
    output_model_params = ' '.join(_output.replace('[', '').replace(']', '').split())

    output = ' '.join((output_results, output_model_params))
    return output


def _get_first_header_line(results, model_params):
    """
    """
    shapes = [results[key].size for key in results.keys()]
    _first_line = '# ' + ' '.join(
        key+'['+str(shape)+']' for key, shape in zip(results.keys(), shapes))
    first_line = _first_line + ' model_params[{0}]'.format(model_params.size) + '\n'
    return first_line
