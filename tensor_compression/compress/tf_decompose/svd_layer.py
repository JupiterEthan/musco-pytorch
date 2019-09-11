"""
Compresses fully-connected layer using TruncatedSVD decomposition.
"""

import tensorflow as tf

from tensorflow.keras import layers


class SVDLayer(layers.Layer):
    """
        This class compute the result using U, S, V -- the result of TruncatedSVD(src_matrix).
        If 'src_matrix' has size M x N then:
        - U has size [..., M, rank]
        - S has size [..., rank, rank]
        - V has size [..., N, rank]
        It adds an original bias vector after x*U*S*V.T computation to the result.
    """

    def __init__(self, src_matrix, bias, rank=None, **kwargs):
        """ Returns a layer that is a result of SVD decomposition of 'src_matrix'.

        :param src_matrix: a kernel of a fully connected layer.
        :param bias: a bias vector.
        :param rank: if it's not None launch TruncatedSVD, apply just SVD otherwise.
        """
        super(SVDLayer, self).__init__(**kwargs)
        s, u, v = tf.linalg.svd(src_matrix, full_matrices=False, compute_uv=True)

        # Truncate decomposition if we need that
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

        # Bias vector from the source fully-connected layer.
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
