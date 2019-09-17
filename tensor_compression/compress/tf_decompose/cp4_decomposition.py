"""CP4 decomposition for Conv2D layer. Replace Conv2D to [Conv2D, DepthwiseConv2D, DepthwiseConv2D, Conv2D]."""
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from sktensor import dtensor, cp_als

from compressor import construct_compressor
from cpd import recompress_ncpd_tensor
from utils import del_keys, to_tf_kernel_order, to_pytorch_kernel_order, depthwise_to_pytorch_kernel_order


def get_conv_params(layer):
    """

    :param layer:
    :return:
    """
    if isinstance(layer, keras.Sequential):
        # If the layer has been decomposed at least once, then
        # the first layer in a sequence contains in_channels,
        # the second layer contains information about kernel_size, padding and strides,
        # the third layer contains information about out_channels.
        layer_1, layer_2, layer_3, layer_4 = layer.layers

        batch_input_shape = layer_1.get_config()['batch_input_shape']
        cin = layer_1.input_shape[-1] if layer_1.data_format == 'channels_last' else layer_1.input_shape[0]
        cout = layer_4.output_shape[-1] if layer_4.data_format == 'channels_last' else layer_4.output_shape[0]

        conf_2, conf_3 = layer_2.get_config(), layer_3.get_config()
        kernel_size = (conf_2['kernel_size'][0], conf_3['kernel_size'][1])
        padding = conf_2['padding']
        strides = (conf_2['strides'][0], conf_3['strides'][1])
    elif isinstance(layer, layers.Conv2D):
        # TODO: this is tha same for CP3 and CP4
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
    :return:
    """
    if isinstance(layer, keras.Sequential):
        w_cin, _, w_h, w_w, w_cout, bias = layer.get_weights()

        w_cin, w_cout = to_pytorch_kernel_order(w_cin), to_pytorch_kernel_order(w_cout)

        # The middle layers are depthwise it should have order
        # [rank, 1, kernel_size, kernel_size]
        # This reorders it correctly from TensorFlow order to PyTorch order
        # w_h, w_w = depthwise_to_pytorch_kernel_order(w_h), depthwise_to_pytorch_kernel_order(w_w)
        w_h, w_w = to_pytorch_kernel_order(w_h), to_pytorch_kernel_order(w_w)

        # TODO: add desc
        w_cin = w_cin.reshape(w_cin.shape[:2]).T
        w_h = w_h.reshape((w_h.shape[0], w_h.shape[2])).T
        w_w = w_w.reshape((w_w.shape[0], w_w.shape[3])).T
        w_cout = w_cout.reshape(w_cout.shape[:2])

        weights = [w_cout, w_cin, w_h, w_w]
    elif isinstance(layer, layers.Conv2D):
        weights, bias = layer.get_weights()
        weights = to_pytorch_kernel_order(weights)

    return weights, bias


def extract_weights_tensors(P):
    w_cin = np.array(P.U[1])
    w_w = np.array(P.U[3] * P.lmbda)
    w_h = np.array(P.U[2])
    w_cout = np.array(P.U[0])

    return w_cin, w_cout, w_h, w_w


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
        w_cout, w_cin, w_h, w_w = recompress_ncpd_tensor(weights,
                                                         new_rank=rank,
                                                         max_cycle=500,
                                                         return_fit=False,
                                                         tensor_format='cpd')
    elif isinstance(layer, layers.Conv2D):
        P, _, _ = cp_als(dtensor(weights), rank, init='random')

        w_cin, w_cout, w_h, w_w = extract_weights_tensors(P)

    # Reshape to proper kernel sizes
    w_h = w_h.T.reshape((rank, 1, kernel_size[0], 1))
    w_w = w_w.T.reshape((rank, 1, 1, kernel_size[1]))
    w_cin = w_cin.T.reshape((rank, cin, 1, 1))
    w_cout = w_cout.reshape([cout, rank, 1, 1])

    # Reorder to TensorFlow order
    w_cin, w_cout = [to_tf_kernel_order(w) for w in [w_cin, w_cout]]

    # The middle layers are depthwise it should have order
    # [rank, 1, kernel_size, kernel_size]
    # This reorders it correctly from TensorFlow order to PyTorch order
    # w_h, w_w = [depthwise_to_pytorch_kernel_order(w) for w in [w_h, w_w]]
    w_h, w_w = [to_tf_kernel_order(w) for w in [w_h, w_w]]

    return [w_cin, w_h, w_w, w_cout], [None, None, None, bias]


def get_layers_params_for_factors(cout, rank, kernel_size, padding, strides, batch_input_shape, **kwargs):
    return [layers.Conv2D, layers.DepthwiseConv2D, layers.DepthwiseConv2D, layers.Conv2D], \
           [{'kernel_size': (1, 1),
             'filters': rank,
             'batch_input_shape': batch_input_shape
             },
            {'kernel_size': (kernel_size[0], 1),
             'padding': padding,
             'strides': (strides[0], 1),
             'use_bias': False,
             },
            {'kernel_size': (1, kernel_size[1]),
             'padding': padding,
             'strides': (1, strides[1]),
             'use_bias': False,
             },
            {'kernel_size': (1, 1),
             'filters': cout}]


def get_config(layer, copy_conf):
    if isinstance(layer, keras.Sequential):
        confs = [l.get_config() for l in layer.layers]
    elif isinstance(layer, layers.Conv2D):
        if copy_conf:
            confs = [layer.get_config()] * 4
        else:
            confs = [{}] * 4

    # New layers have other 'units', 'kernel_initializer', 'bias_initializer' and 'name'.
    # That's why we delete them to prevent double definition.
    redundant_keys = {'kernel_initializer', 'bias_initializer', 'name', 'kernel_size', 'padding', 'strides', 'filters'}
    for conf_idx, _ in enumerate(confs):
        confs[conf_idx] = del_keys(confs[conf_idx], redundant_keys)
    return confs


get_cp4_seq = construct_compressor(get_conv_params,
                                   get_cp_factors,
                                   get_layers_params_for_factors,
                                   get_config,
                                   (layers.Conv2D, keras.Sequential))
