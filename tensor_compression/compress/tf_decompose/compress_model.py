"""
This module contains functions for compressing fully-connected and conv layers.
"""
import re

from absl import logging
from tensorflow.keras.models import Model
from tensorflow import keras
# from svd_layer import get_svd_seq, SVDLayer
from cp3_decomposition import get_cp3_seq
from cp4_decomposition import get_cp4_seq
from svd_decomposition import get_svd_seq
from tucker2_decomposition import get_tucker2_seq


def get_compressed_sequential(model, decompose_info, optimize_rank=False, vbmf=True, vbmf_weaken_factor=0.8):
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
                                        optimize_rank=optimize_rank,
                                        vbmf=vbmf,
                                        vbmf_weaken_factor=vbmf_weaken_factor)
        else:
            logging.info('Incorrect decomposition type for the layer {}'.format(layer.name))
            raise NameError("Wrong Decomposition Name. You should use one of: ['svd', 'cp3', 'cp4', 'tucker-2']")

        x = new_layer(x)
        new_model.add(new_layer)

    return new_model


def insert_layer_nonseq(model, layer_regexs,
                        insert_layer_name=None, position='after'):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name

            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    last_layer = model.layers[0]
    # Iterate over all layers after the input

    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        changed = False
        for layer_regex, new_layer in layer_regexs.items():
            if not re.match(layer_regex, layer.name):
                continue

            changed = True
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            x = new_layer(x)
            last_layer = new_layer
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            if position == 'before':
                x = layer(x)
                last_layer = layer
            break
        if not changed:
            x = layer(layer_input)
            last_layer = layer

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=last_layer.output)


def get_compressed_model(model, decompose_info, optimize_rank=False, vbmf=True, vbmf_weaken_factor=0.8):
    new_model = model
    changed = False

    layer_regexs = dict()
    for idx, layer in enumerate(model.layers[1:]):
        if layer.name not in decompose_info:
            continue

        # from pathlib import Path

        # model_file = Path("./test.h5")
        # if model_file.exists() and changed:
        #     new_model.load_weights("./test.h5")

        decompose, decomp_rank = decompose_info[layer.name]

        insert_layer_factory = None
        if decompose.lower() == 'svd':
            insert_layer_factory = get_svd_seq(layer, rank=decomp_rank)
        elif decompose.lower() == 'cp3':
            insert_layer_factory = get_cp3_seq(layer,
                                                       rank=decomp_rank,
                                                       optimize_rank=optimize_rank)
        elif decompose.lower() == 'cp4':
            insert_layer_factory = get_cp4_seq(layer,
                                                       rank=decomp_rank,
                                                       optimize_rank=optimize_rank)
        elif decompose.lower() == 'tucker2':
            layer_regexs[layer.name] = get_tucker2_seq(layer,
                                                           rank=decomp_rank,
                                                           optimize_rank=optimize_rank,
                                                           vbmf=vbmf,
                                                           vbmf_weaken_factor=vbmf_weaken_factor)

    new_model = insert_layer_nonseq(new_model, layer_regexs,
                                    insert_layer_name=None, position='replace')

    return new_model
