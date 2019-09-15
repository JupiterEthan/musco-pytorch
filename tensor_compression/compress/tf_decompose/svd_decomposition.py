import numpy as np

from compressor import construct_compressor
from tensorflow import keras
from tensorflow.keras import layers

from utils import del_keys


def get_params(layer):
    params = dict()
    if isinstance(layer, keras.Sequential):
        u, sv_adj = layer.layers
        weights_shape = (u.get_weights()[0].shape[0], sv_adj.get_weights()[0].shape[-1])
    else:
        weights_shape = layer.get_weights()[0].shape
    params['weights_shape'] = weights_shape
    return params


def get_svd_factors(layer, rank, **kwargs):
    if isinstance(layer, keras.Sequential):
        U, U_b, SV_adj, SV_adj_bias = layer.get_weights()

        u, s, v_adj = np.linalg.svd(U, full_matrices=False)

        # Truncate ranks
        u = u[..., :rank]
        s = np.diag(s[..., :rank])
        v_adj = v_adj.T[..., :rank].T

        return [u, np.dot(np.dot(s, v_adj), SV_adj)], [None, SV_adj_bias]
    elif isinstance(layer, layers.Dense):
        weights, bias = layer.get_weights()

        u, s, v_adj = np.linalg.svd(weights, full_matrices=False)

        # If rank is None take the original rank of weights.
        rank = min(weights.shape) if rank is None else rank

        # Truncate ranks
        u = u[..., :rank]
        s = np.diag(s[..., :rank])
        v_adj = v_adj.T[..., :rank].T

        return [u, np.dot(s, v_adj)], [None, bias]


def get_layers_params_for_factors(rank, weights_shape, **kwargs):
    return [layers.Dense, layers.Dense], [{'units': rank},
                                          {'units': weights_shape[-1]}]


def get_config(layer, copy_conf):
    confs = []

    # New layers have other 'units', 'kernel_initializer', 'bias_initializer' and 'name'.
    # That's why we want to delete them from confs to prevent double definition.
    redundant_keys = {'units', 'kernel_initializer', 'bias_initializer', 'name'}

    if isinstance(layer, keras.Sequential):
        for l in layer.layers:
            confs.append(del_keys(l.get_config(), redundant_keys))
    elif isinstance(layer, layers.Dense):
        # Get conf of the source layer
        conf = {} if not copy_conf else del_keys(layer.get_config(), redundant_keys)

        # Source layer is decomposed into 3, that's why we need 3 confs here.
        confs = [conf] * 3

    return confs


get_svd_seq = construct_compressor(get_params,
                                   get_svd_factors,
                                   get_layers_params_for_factors,
                                   get_config,
                                   (layers.Dense, keras.Sequential))
