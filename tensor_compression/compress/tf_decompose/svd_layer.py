"""
Uses the result of Truncated SVD decomposition as weights instead of a src_matrix.
"""

import tensorflow as tf

from tensorflow.keras import layers


class SVDLayer(layers.Layer):
    """
        This class returns a layer that contains of three sequential matrices U, S, V.
        These matrices are the result of TruncatedSVD decomposition of 'src_matrix'.
    """

    def __init__(self, src_matrix, bias, rank=None):
        """ Returns a layer that is a result of SVD decomposition of the source one.

        :param src_matrix: tensor of a fully connected layer.
        :param bias: bias vector.
        :param rank: if it's not None launch truncated SVD, apply just SVD otherwise.
        """
        super(SVDLayer, self).__init__()
        s, u, v = tf.linalg.svd(src_matrix, full_matrices=False, compute_uv=True)

        # Leave only first 'rank' singular values (Truncated SVD)
        if rank is not None:
            u = u[..., :rank]
            s = s[..., :rank]
            v = v[..., :rank]

        s = tf.linalg.diag(s)

        # This variables will automatically be added to self.weights
        # in the order they are added below.
        # Refer https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models#the_layer_class for details.
        self.u = tf.Variable(initial_value=u, name='U')
        self.s = tf.Variable(initial_value=s, name='S')
        self.v = tf.Variable(initial_value=v, name='V')

        self.bias = tf.Variable(initial_value=bias, name="bias")

    def call(self, inputs, **kwargs):
        x = tf.matmul(inputs, self.u)
        x = tf.matmul(x, self.s)
        x = tf.matmul(x, self.v, adjoint_b=True)
        x = x + self.bias
        return x

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.v.get_shape()[0]
        return tuple(output_shape)
