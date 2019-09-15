import numpy as np

from compressor import construct_compressor
from tensorflow import keras
from tensorflow.keras import layers


def get_params(layer):
    return {'weights_shape': layer.get_weights()[0].shape}  # TODO edit sequential


def get_svd_factors(layer, rank, **kwargs):
    weights, bias = layer.get_weights()  # add sequential case

    u, s, v_adj = np.linalg.svd(weights, full_matrices=False)

    # If rank is None take the original rank of weights.
    rank = min(weights.shape) if rank is None else rank

    # Truncate ranks
    u = u[..., :rank]
    s = np.diag(s[..., :rank])
    v_adj = v_adj.T[..., :rank].T

    return [u, s, v_adj], [None, None, bias]


def get_layers_params_for_factors(rank, weights_shape, **kwargs):
    return [layers.Dense, layers.Dense, layers.Dense], [{'units': rank,
                                                         # 'batch_input_shape': batch_input_shape
                                                         },
                                                        {
                                                            'units': rank
                                                        },
                                                        {
                                                            'units': weights_shape[-1]
                                                        }
                                                        ]


def get_config(layer, copy_conf):
    # Get conf of the source layer
    confs = {}
    if copy_conf:
        confs = layer.get_config()

        # New layers have other 'units', 'kernel_initializer', 'bias_initializer' and 'name'.
        # That's why we delete them to prevent double definition.
        del confs['units'], confs['kernel_initializer'], confs['bias_initializer'], confs['name']
    return [confs]*3 # TODO: fix for sequential


get_svd_seq = construct_compressor(get_params,
                                   get_svd_factors,
                                   get_layers_params_for_factors,
                                   get_config,
                                   (layers.Dense, keras.Sequential))
