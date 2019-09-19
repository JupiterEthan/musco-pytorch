"""General utils for layer compressors."""

import numpy as np


def to_tf_kernel_order(tensor):
    """Change conv.kernel axis order from PyTorch to Tensoflow.

    :param tensor: tensor with conv.kernel weights.
    :return: tensor with the Tensoflow-like exis order.
    []
    """
    return np.transpose(tensor, (2, 3, 1, 0))


def to_pytorch_kernel_order(tensor):
    """Change conv.kernel axis order from Tensoflow to PyTorch.

    :param tensor: tensor with conv.kernel weights.
    :return: tensor with the Pytorch-like exis order.
    []
    """
    return np.transpose(tensor, (3, 2, 0, 1))


def del_keys(src_dict, del_keys):
    """Deletes redundant_keys from conf.

    :param src_dict: a dict
    :param del_keys: a list/set/etc with key names that we want to delete.
    :return: the copy of dict without keys from del_keys.
    """
    return {key: value for key, value in src_dict.items() if key not in del_keys}