"""
This module contains functions for compressing fully-connected and conv layers.
"""

from absl import logging
from tensorflow import keras

from svd_layer import SVDLayer


def get_compressed_model(model, decompose_info):
    """Compresses source model using decompositions from decompose_info dict.

    For example if decompose_info = {
            'dense': ('svd', 10)
    }
    it means that the layer with the name 'dense' will be compressed
    using TruncatedSVD with truncation rank 10.

    For fully-connected layer you can use SVD decomposition
    For convolution layer networks CP3, CP4, Tucker-2 are available.

    :param model: source model.
    :param decompose_info: dict that describes what layers compress using what decomposition method.
                           Possible decompositions are: 'svd', 'cp3', 'cp4', 'tucker-2'.
    :return: new tf.keras.Model with compressed layers.
    """
    model_input = model.input
    x = model_input

    for idx, layer in enumerate(model.layers):
        new_layer = layer
        if layer.name in decompose_info:
            decompose, decomp_rank = decompose_info[layer.name]
            if decompose.lower() == 'svd':
                logging.info('SVD layer {}'.format(layer.name))
                new_layer = SVDLayer(*layer.get_weights(), rank=decomp_rank)
            else:
                logging.info('Incorrect decomposition type for the layer {}'.format(layer.name))

        x = new_layer(x)

    return keras.Model(model_input, x)


