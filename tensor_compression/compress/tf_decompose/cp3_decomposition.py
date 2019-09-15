"""
Compresses layers.Conv2D layer using CP3.
"""
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from sktensor import dtensor, cp_als

from compressor import construct_compressor
from cpd import recompress_ncpd_tensor
from utils import to_tf_kernel_order, to_pytorch_kernel_order


def get_conv_params(layer):
    """

    :param layer:
    :return:
    """
    if isinstance(layer, keras.Sequential):
        if any(layer.data_format != 'channels_last' for layer in layer.layers):
            raise Exception("channel_last is only supported.")

        # If the layer has been decomposed at least once, then
        # the first layer in a sequence contains in_channels,
        # the second layer contains information about kernel_size, padding and strides,
        # the third layer contains information about out_channels.
        conf_conv_2 = layer.layers[1].get_config()

        kernel_size = conf_conv_2['kernel_size']
        padding = conf_conv_2['padding']
        strides = conf_conv_2['strides']

        first_layer = layer.layers[0]
        last_layer = layer.layers[-1]
        cin = first_layer.input_shape[-1] if first_layer.data_format == 'channels_last' else first_layer.input_shape[0]
        cout = last_layer.output_shape[-1] if last_layer.data_format == 'channels_last' else last_layer.output_shape[0]

        batch_input_shape = first_layer.get_config()['batch_input_shape']

    elif isinstance(layer, layers.Conv2D):
        cin = layer.input_shape[-1] if layer.data_format == 'channels_last' else layer.input_shape[0]
        cout = layer.output_shape[-1] if layer.data_format == 'channels_last' else layer.output_shape[0]
        layer_conf = layer.get_config()
        kernel_size = layer_conf['kernel_size']
        padding = layer_conf['padding']
        strides = layer_conf['strides']
        batch_input_shape = layer_conf['batch_input_shape']

    return {'cin': cin,
            'cout': cout,
            'kernel_size': kernel_size,
            'padding': padding,
            'strides': strides,
            'batch_input_shape': batch_input_shape}


def get_weights_and_bias(layer):
    """Returns weights and biases.

    :param layer: a source layer
    :return: If layer is tf.keras.layers.Conv2D layer.weights is returned as weights,
             Otherwise a list of weight tensors and bias tensor are returned as weights.
             The second element that is returned is a bias tensor.
             Note that all weights are returned in PyTorch dimension order:
             [out_channels, in_channels, kernel_size[0]*kernel_size[1]]
    """
    if isinstance(layer, keras.Sequential):
        w_cin, _ = layer.layers[0].get_weights()
        w_z = layer.layers[1].get_weights()[0]
        w_cout, bias = layer.layers[2].get_weights()

        w_cin, w_cout = [to_pytorch_kernel_order(w) for w in [w_cin, w_cout]]

        # The middle layer is depthwise it should have order
        # [rank, 1, kernel_size, kernel_size]
        # This reorders it correctly from TensorFlow order to PyTorch order
        w_z = np.transpose(w_z, (2, 3, 0, 1))

        # Reshape 4D tensors into 4D matrix.
        # w_cin and w_cout have two dimension of size 1.
        # w_z has second dimension that is equal to 1.
        w_cin = w_cin.reshape(w_cin.shape[:2]).T
        w_cout = w_cout.reshape(w_cout.shape[:2])
        w_z = w_z.reshape((w_z.shape[0], np.prod(w_z.shape[2:]))).T

        weights = [w_cout, w_cin, w_z]
    elif isinstance(layer, layers.Conv2D):
        weights, bias = layer.get_weights()
        weights = to_pytorch_kernel_order(weights)
        weights = weights.reshape((*weights.shape[:2], -1))

    return weights, bias


def get_cp_factors(layer, rank, cin, cout, kernel_size, **kwargs):
    """

    :param layer:
    :param weights:
    :param bias:
    :param rank:
    :param cin:
    :param cout:
    :param kernel_size:
    :param kwergs:
    :return:
    """
    weights, bias = get_weights_and_bias(layer)

    if isinstance(layer, keras.Sequential):
        w_cout, w_cin, w_z = recompress_ncpd_tensor(weights,
                                                    new_rank=rank,
                                                    max_cycle=500,
                                                    return_fit=False,
                                                    tensor_format='cpd')
    elif isinstance(layer, layers.Conv2D):
        P, _, _ = cp_als(dtensor(weights), rank, init='random')

        w_cin, w_cout, w_z = extract_weights_tensors(P)

    # Reshape to the proper PyTorch shape order
    w_cin = w_cin.T.reshape((rank, cin, 1, 1))
    w_z = w_z.T.reshape((rank, 1, *kernel_size))
    w_cout = w_cout.reshape((cout, rank, 1, 1))

    # Reorder to TensorFlow order
    w_cin, w_cout = [to_tf_kernel_order(w) for w in [w_cin, w_cout]]

    # The middle layer is depthwise it should have order
    # [rank, 1, kernel_size, kernel_size]
    # This reorders it correctly from TensorFlow order to PyTorch order
    w_z = np.transpose(w_z, (2, 3, 0, 1))

    return [w_cin, w_z, w_cout], [None, None, bias]


def extract_weights_tensors(P):
    w_cout = np.array(P.U[0])
    w_cin = np.array(P.U[1])
    w_z = (np.array(P.U[2]) * (P.lmbda))
    return w_cin, w_cout, w_z


def get_layers_params_for_factors(cout, rank, kernel_size, padding, strides, batch_input_shape, **kwargs):
    return [layers.Conv2D, layers.DepthwiseConv2D, layers.Conv2D], [{'kernel_size': (1, 1),
                                                                     'filters': rank,
                                                                     'batch_input_shape': batch_input_shape
                                                                     },
                                                                    {'kernel_size': kernel_size,
                                                                     'padding': padding,
                                                                     'strides': strides,
                                                                     'use_bias': False,
                                                                     },
                                                                    {'kernel_size': (1, 1),
                                                                     'filters': cout}]


def get_config(layer, copy_conf):
    if isinstance(layer, keras.Sequential):
        confs = [l.get_config() for l in layer.layers]
    elif isinstance(layer, layers.Conv2D):
        if copy_conf:
            confs = [layer.get_config()]*3
        else:
            confs = [{}]*3

    # New layers have other 'units', 'kernel_initializer', 'bias_initializer' and 'name'.
    # That's why we delete them to prevent double definition.
    for conf_idx, _ in enumerate(confs):
        for key in ['kernel_initializer', 'bias_initializer', 'name', 'kernel_size', 'padding', 'strides', 'filters']:
            if key not in confs[conf_idx]:
                continue
            del confs[conf_idx][key]

    return confs


get_cp3_seq = construct_compressor(get_conv_params,
                                   get_cp_factors,
                                   get_layers_params_for_factors,
                                   get_config,
                                   (layers.Conv2D, keras.Sequential))
