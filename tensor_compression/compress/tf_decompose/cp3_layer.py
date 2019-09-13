"""
Compresses layers.Conv2D layer using CP3.
"""
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sktensor import dtensor, cp_als

from cpd import recompress_ncpd_tensor


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

    return cin, cout, kernel_size, padding, strides


def to_tf_kernel_order(tensor):
    """Change conv.kernel axis order from PyTorch to Tensoflow.

    :param tensor: tensor with conv.kernel weights.
    :return: tensor with the Tensoflow-like exis order.
    """
    return np.transpose(tensor, (2, 3, 1, 0))


def to_pytorch_kernel_order(tensor):
    """Change conv.kernel axis order from Tensoflow to PyTorch.

    :param tensor: tensor with conv.kernel weights.
    :return: tensor with the Pytorch-like exis order.
    """
    return np.transpose(tensor, (3, 2, 0, 1))


def get_weights_and_bias(layer):
    """Returns weights and biases.

    :param layer: a source layer
    :return: If layer is tf.keras.layers.Conv2D layer.weights is returned as weights,
             Otherwise a list of weight tensors and bias tensor are returned as weights.
             The second element that is returned is a bias tensor.
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
        print("!!!!!!", weights.shape)
        # weights = weights.T
        weights = to_pytorch_kernel_order(weights)
        weights = weights.reshape((*weights.shape[:2], -1))

    return weights, bias


def get_cp_factors(layer, weights, bias, rank, cin, cout, kernel_size):
    """

    :param layer:
    :param weights:
    :param bias:
    :param rank:
    :param cin:
    :param cout:
    :param kernel_size:
    :return:
    """
    check_layer(layer)

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

    # Reshape to the proper results for a conv layer
    # [output_c, input_c, h, w]
    # [h, w, input_channel, out_channel]
    w_cin = w_cin.T.reshape((rank, cin, 1, 1))
    w_z = w_z.T.reshape((rank, 1, *kernel_size))
    w_cout = w_cout.reshape((cout, rank, 1, 1))

    return [w_cin, w_z, w_cout], [None, None, bias]


def get_layers_for_factors(cout, rank, kernel_size, padding, strides):
    return [layers.Conv2D, layers.DepthwiseConv2D, layers.Conv2D], [{'kernel_size': (1, 1),
                                                                     'filters': rank,
                                                                     },
                                                                    {'kernel_size': kernel_size,
                                                                     'padding': padding,
                                                                     'strides': strides,
                                                                     'use_bias': False,
                                                                     },
                                                                    {'kernel_size': (1, 1),
                                                                     'filters': cout}]
    # return [layers.Conv2D, layers.Conv2D, layers.Conv2D], [{'kernel_size': (1, 1),
    #                                                         'filters': rank,
    #                                                         },
    #                                                        {'kernel_size': kernel_size,
    #                                                         'padding': padding,
    #                                                         'strides': strides,
    #                                                         'filters': rank,
    #                                                         },
    #                                                        {'kernel_size': (1, 1),
    #                                                         'filters': cout}]


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


def build_sequence(layer, weights, biases, layer_classes, layer_confs, conf):
    layer_seq = keras.Sequential(name=layer.name)
    for idx, (weight, bias, layer_class, layer_conf) in enumerate(zip(weights, biases, layer_classes, layer_confs)):
        # print(layer_class.__name__)
        if layer_class.__name__ == 'DepthwiseConv2D':
            # [batch, out_height, out_width, in_channels * channel_multiplier]
            # weight = np.transpose(weight,  (2, 3, 0, 1))
            weight = to_tf_kernel_order(weight)
            new_layer = layers.DepthwiseConv2D(name="{}-{}".format(layer.name, idx),
                                               **layer_conf,
                                               # kernel_initializer=tf.constant_initializer(weight),
                                               )
        else:
            # weight = np.transpose(weight,  (2, 3, 0, 1))
            weight = to_tf_kernel_order(weight)
            new_layer = layer_class(name="{}-{}".format(layer.name, idx),
                                    # kernel_initializer=tf.constant_initializer(weight),
                                    bias_initializer='zeros' if idx == 0 else tf.constant_initializer(bias),
                                    **layer_conf,
                                    **conf)
        layer_seq.add(new_layer)
    return layer_seq


def get_cp3_seq(layer, rank, pretrained=None, copy_conf=False):
    """

    :param layer:
    :param rank:
    :param pretrained:
    :return:
    """
    cin, cout, kernel_size, padding, stride = get_conv_params(layer)

    weights, bias = get_weights_and_bias(layer)
    weights, biases = get_cp_factors(layer, weights, bias, rank, cin, cout, kernel_size)

    cp3_seq = build_sequence(layer,
                             weights,
                             biases,
                             *get_layers_for_factors(cout, rank, kernel_size, padding, stride),
                             get_config(layer))

    return cp3_seq


class LayerDecomposer():
    def __init__(self, accepted_layers):
        self.accepted_layers = accepted_layers

    def __call__(self, *args, **kwargs):
        pass


class CP3LayerDecomposer(LayerDecomposer):
    def __init__(self):
        super(CP3LayerDecomposer, self).__init__()
        pass

    def __call__(self, *args, **kwargs):
        pass
