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


def check_layer_type(layer):
    return isinstance(layer, (keras.Sequential, layers.Conv2D))


def check_data_format(layer):
    if isinstance(layer, keras.Sequential):
        return any(layer.data_format != 'channel_last' for layer in layer.layers)
    elif isinstance(layer, layers.Conv2D):
        return layer.data_format != 'channel_last'


def check_layer(layer):
    if not check_layer_type(layer):
        raise TypeError("This function in only applicable to "
                        "{} or {}. "
                        "But this one is {}".format(keras.Sequential.__name__,
                                                    layers.Conv2D.__name__,
                                                    type(layer)))
    if not check_data_format(layer):
        raise Exception("channel_last is only supported but "
                        "current model has {}. ".format(layer.data_format))


def get_conv_params(layer):
    """

    :param layer:
    :return:
    """
    check_layer(layer)

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
    elif isinstance(layer, layers.Conv2D):
        cin = layer.input_shape[-1] if layer.data_format == 'channels_last' else layer.input_shape[0]
        cout = layer.output_shape[-1] if layer.data_format == 'channels_last' else layer.output_shape[0]
        layer_conf = layer.get_config()
        kernel_size = layer_conf['kernel_size']
        padding = layer_conf['padding']
        strides = layer_conf['strides']

    return {'cin': cin,
            'cout': cout,
            'kernel_size': kernel_size,
            'padding': padding,
            'strides': strides}


def get_weights_and_bias(layer):
    """Returns weights and biases.

    :param layer: a source layer
    :return: If layer is tf.keras.layers.Conv2D layer.weights is returned as weights,
             Otherwise a list of weight tensors and bias tensor are returned as weights.
             The second element that is returned is a bias tensor.
             Note that all weights are returned in PyTorch dimension order:
             [out_channels, in_channels, kernel_size[0]*kernel_size[1]]
    """
    check_layer(layer)
    if isinstance(layer, keras.Sequential):
        w_cin, bias_cin = to_pytorch_kernel_order(layer.layers[0].get_weights())
        w_z, bias_z = to_pytorch_kernel_order(layer.layers[1].get_weights())
        w_cout, bias_cout = to_pytorch_kernel_order(layer.layers[2].get_weights())

        bias = bias_cout

        # Reshape 4D tensors into 4D matrix.
        # w_cin and w_cout have two dimension of size 1.
        # w_z has second dimension that is equal to 1.
        w_z = w_z.reshape((w_z.shape[0], np.prod(w_z.shape[2:]))).T
        w_cin = np.squeeze(w_cin).T
        w_cout = np.squeeze(w_cout)

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
    check_layer(layer)

    weights, bias = get_weights_and_bias(layer)

    if isinstance(layer, keras.Sequential):
        w_cout, w_cin, w_z = recompress_ncpd_tensor(weights,
                                                    new_rank=rank,
                                                    max_cycle=500,
                                                    return_fit=False,
                                                    tensor_format='cpd')
    elif isinstance(layer, layers.Conv2D):
        P, _, _ = cp_als(dtensor(weights), rank, init='random')

        w_cout = np.array(P.U[0])
        w_cin = np.array(P.U[1])
        w_z = (np.array(P.U[2]) * (P.lmbda))

    # Reshape to the proper PyTorch shape order
    w_cin = w_cin.T.reshape((rank, cin, 1, 1))
    w_z = w_z.T.reshape((rank, 1, *kernel_size))
    w_cout = w_cout.reshape((cout, rank, 1, 1))

    # Reorder to TensorFlow order
    w_cin, w_z, w_cout = [to_tf_kernel_order(w) for w in [w_cin, w_z, w_cout]]

    return [w_cin, w_z, w_cout], [None, None, bias]


def get_layers_params_for_factors(cout, rank, kernel_size, padding, strides, **kwargs):
    # TODO: return another result for sequence
    return [layers.Conv2D, layers.Conv2D, layers.Conv2D], [{'kernel_size': (1, 1),
                                                            'filters': rank,
                                                            },
                                                           {'kernel_size': kernel_size,
                                                            'padding': padding,
                                                            'strides': strides,
                                                            'filters': rank,
                                                            },
                                                           {'kernel_size': (1, 1),
                                                            'filters': cout}]


def get_config(layer):
    check_layer(layer)

    if isinstance(layer, keras.Sequential):
        conf = layer.layers[0].get_config()
    elif isinstance(layer, layers.Conv2D):
        conf = layer.get_config()

    # New layers have other 'units', 'kernel_initializer', 'bias_initializer' and 'name'.
    # That's why we delete them to prevent double definition.
    del conf['kernel_initializer'], conf['bias_initializer'], conf['name']
    del conf['kernel_size'], conf['padding'], conf['strides'], conf['filters']

    return conf


get_cp3_seq = construct_compressor(get_conv_params, get_cp_factors, get_layers_params_for_factors, get_config)
