"""
This module contains functions for compressing fully-connected and conv layers.
"""

from absl import logging
from tensorflow import keras

# from svd_layer import get_svd_seq, SVDLayer
from cp3_decomposition import get_cp3_seq
from cp4_decomposition import get_cp4_seq
from svd_decomposition import get_svd_seq
from tucker2_decomposition import get_tucker2_seq


def get_compressed_model(model, decompose_info, optimize_rank=False):
    """Compresses source model using decompositions from decompose_info dict.

    For example if decompose_info = {
            'dense': ('svd', 10)
    }
    it means that the layer with the name 'dense' will be compressed
    using TruncatedSVD with truncation rank 10.

    For fully-connected layer you can use SVD decomposition
    For convolution layer networks CP3, CP4, Tucker-2 are available.

    If you want learn more about different tensor decomposition refer:

    'Tensor Networks for Dimensionality Reduction and Large-Scale Optimization.
    Part 1 Low-Rank Tensor Decompositions.'

    :param model: source model.
    :param decompose_info: dict that describes what layers compress using what decomposition method.
                           Possible decompositions are: 'svd', 'cp3', 'cp4', 'tucker-2'.
    :return: new tf.keras.Model with compressed layers.
    """
    x = model.input
    new_model = keras.Sequential([])
    for idx, layer in enumerate(model.layers):
        if layer.name not in decompose_info:
            x = layer(x)
            new_model.add(layer)
            continue

        decompose, decomp_rank = decompose_info[layer.name]
        if decompose.lower() == 'svd':
            logging.info('SVD layer {}'.format(layer.name))
            new_layer = get_svd_seq(layer, rank=decomp_rank)
        elif decompose.lower() == 'cp3':
            logging.info('CP3 layer {}'.format(layer.name))
            new_layer = get_cp3_seq(layer,
                                    rank=decomp_rank,
                                    optimize_rank=optimize_rank)
        elif decompose.lower() == 'cp4':
            logging.info('CP4 layer {}'.format(layer.name))
            new_layer = get_cp4_seq(layer,
                                    rank=decomp_rank,
                                    optimize_rank=optimize_rank)
        elif decompose.lower() == 'tucker2':
            logging.info('Tucker2 layer {}'.format(layer.name))
            new_layer = get_tucker2_seq(layer,
                                        rank=decomp_rank,
                                        optimize_rank=optimize_rank)
        else:
            logging.info('Incorrect decomposition type for the layer {}'.format(layer.name))
            raise NameError("Wrong Decomposition Name. You should use one of: ['svd', 'cp3', 'cp4', 'tucker-2']")

        x = new_layer(x)
        new_model.add(new_layer)

    return new_model

