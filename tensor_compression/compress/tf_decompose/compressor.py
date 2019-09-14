""""""
import tensorflow as tf

from tensorflow import keras


def build_sequence(layer, weights, biases, layer_classes, layer_confs, conf):
    layer_seq = keras.Sequential(name=layer.name)
    for idx, (weight, bias, layer_class, layer_conf) in enumerate(zip(weights, biases, layer_classes, layer_confs)):
        new_layer = layer_class(name="{}-{}".format(layer.name, idx),
                                kernel_initializer=tf.constant_initializer(weight),
                                bias_initializer='zeros' if bias is None else tf.constant_initializer(bias),
                                **layer_conf,
                                **conf)
        layer_seq.add(new_layer)
    return layer_seq


def construct_compressor(get_params, get_decomposer, get_factor_params, get_config):
    """The protopyte of generator.

    :param get_params:
    :param get_decomposer:
    :param get_factor_params:
    :return:
    """
    def compressor(layer, rank, pretrained=None, copy_conf=False, **kwargs):
        """

        :param layer:
        :param ranks:
        :param kwargs:
        :return:
        """
        params = get_params(layer)
        weights, biases = get_decomposer(layer, rank, **params)
        params_for_factors = get_factor_params(rank=rank, **params)
        layer_seq = build_sequence(layer,
                                   weights,
                                   biases,
                                   *params_for_factors,
                                   get_config(layer) if copy_conf else {})
        return layer_seq

    return compressor