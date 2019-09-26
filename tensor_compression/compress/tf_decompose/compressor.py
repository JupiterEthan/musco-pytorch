""""""
import tensorflow as tf

from tensorflow import keras


def check_layer_type(layer, accepted_layers):
    return isinstance(layer, accepted_layers)


def check_data_format(layer):
    if isinstance(layer, keras.Sequential):
        return any(layer.data_format != 'channel_last' for layer in layer.layers)
    else:
        return layer.data_format != 'channel_last'


def check_layer(layer, accepted_layers):
    if not check_layer_type(layer, accepted_layers):
        raise TypeError("This function in only applicable for "
                        "{}. "
                        "But this one is {}".format(", ".join([acc_layer.__name__ for acc_layer in accepted_layers]),
                                                    type(layer)))
    if not check_data_format(layer):
        raise Exception("channel_last is only supported but "
                        "current model has {}. ".format(layer.data_format))


def del_redundant_keys(dict_1, dict_2):
    del_keys = []
    for key in dict_1:
        if key not in dict_2:
            continue
        del_keys.append(key)

    for key in del_keys:
        del dict_1[key]

    return dict_1


def build_sequence(layer, weights, biases, layer_classes, layer_confs, confs):
    layer_seq = keras.Sequential(name=layer.name)
    for idx, (weight, bias, layer_class, layer_conf, conf) in enumerate(zip(weights, biases, layer_classes, layer_confs, confs)):
        conf = del_redundant_keys(conf, layer_conf)
        new_layer = layer_class(name="{}-{}".format(layer.name, idx),
                                kernel_initializer=tf.constant_initializer(weight),
                                bias_initializer='zeros' if bias is None else tf.constant_initializer(bias),
                                **layer_conf,
                                **conf)
        layer_seq.add(new_layer)
    return layer_seq


def construct_compressor(get_params, get_rank, get_decomposer, get_factor_params, get_config, accepted_layers):
    """The protopyte of generator.

    :param get_params:
    :param get_decomposer:
    :param get_factor_params:
    :return:
    """

    def compressor(layer, rank, optimize_rank=False, copy_conf=False, **kwargs):
        """

        :param layer:
        :param ranks:
        :param kwargs:
        :return:
        """
        check_layer_type(layer, accepted_layers)
        params = get_params(layer)
        if optimize_rank and get_rank is not None:
            rank = get_rank(layer, rank=rank, **params, **kwargs)
            rank = rank if rank != 0 else 1
            print("!!!!!", rank)
        weights, biases = get_decomposer(layer, rank, **params)
        params_for_factors = get_factor_params(rank=rank, **params)
        layer_seq = build_sequence(layer,
                                   weights,
                                   biases,
                                   *params_for_factors,
                                   get_config(layer, copy_conf))
        return layer_seq

    return compressor
