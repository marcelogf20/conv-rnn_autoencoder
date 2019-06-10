"""" File containing custom operations and layers for keras models """

import numpy as np
import tensorflow as tf


class Binarize(tf.keras.layers.Layer):
    """ Keras layer that implements Toderici's binarization strategy """
    def __init__(self, low=0, high=1, name=None):
        self.low = tf.constant(low, tf.float32)
        self.high = tf.constant(high, tf.float32)
        super(Binarize, self).__init__(name=name)

    def build(self, input_shape):
        super(Binarize, self).build(input_shape)

    def call(self, inputs):
        tensor, is_evaluation = inputs[0], inputs[1]
        if is_evaluation:
            op = self.binarize
        else:
            op = self.random_bin

        output = op(tensor, self.high, self.low)
        return output

    def compute_output_shape(self, input_shape):
        tensor_shape, flag_shape = input_shape[0], input_shape[1]
        return tensor_shape

    @staticmethod
    @tf.custom_gradient
    def random_bin(tensor, high, low):
        """ Random binarization that calculates the input_array probability
            of becoming one of the binary code elements by an amount
            proportional to it's distance from the binary element
        """
        code_len = tf.abs(high - low)
        prob_high = (tensor - low) / code_len
        bernoulli = tf.distributions.Bernoulli(probs=prob_high)
        output = tf.cast(bernoulli.sample(), dtype=tf.float32)

        def grad(dy):
            return dy, None, None

        return output, grad

    @staticmethod
    @tf.custom_gradient
    def binarize(tensor, high, low):
        """ Routine to binarize an input considering a code of two numbers.
            The binarization transform the tensor based on the minimum
            length from elements of the binary code.
        """
        dist_to_high = tf.abs(high - tensor)
        dist_to_low = tf.abs(tensor - low)
        dist_tensor = tf.stack([dist_to_low, dist_to_high])
        near_high = tf.cast(tf.argmin(dist_tensor, axis=0),
                            dtype=tf.float32)
        near_low = tf.abs(near_high - 1.)
        output = low * near_low + high * near_high

        def grad(dy):
            return dy, None, None

        return output, grad


class Quantize(tf.keras.layers.Layer):
    """ Keras layer representing quantization. In training time it
        simply puts random noise on data. On test time it's made a simple
        round operation
    """
    def __init__(self, factor=1, name=None):
        self.factor = tf.constant(factor, dtype=tf.float32)
        self.half = tf.constant(.5, dtype=tf.float32)
        super(Quantize, self).__init__(name=name)

    def build(self, input_shape):
        super(Quantize, self).build(input_shape)

    def call(self, inputs):
        tensor, is_evaluating = inputs
        tensor *= self.factor
        if is_evaluating:
            output = tf.math.round(tensor)
        else:
            noise = tf.random.uniform(tf.shape(tensor), -self.half,
                                      self.half)
            output = tensor + noise

        return output

    def compute_output_shape(self, input_shape):
        tensor_shape, flag_shape = input_shape
        return tensor_shape


class ExpandDims(tf.keras.layers.Layer):
    """ Wrapper for the expand dims operation of keras backend """
    def __init__(self, axis=1, name=None):
        self.axis = axis
        super(ExpandDims, self).__init__(name=name)

    def build(self, input_shape):
        super(ExpandDims, self).build(input_shape)

    def call(self, tensor):
        output = tf.keras.backend.expand_dims(tensor, self.axis)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape = output_shape[:self.axis] + [1] + \
            output_shape[self.axis:]
        output_shape = tf.TensorShape(output_shape)
        return output_shape


class GetOnes(tf.keras.layers.Layer):
    """ Auxiliary layer that retunrs one's of the same shape of input """
    def __init__(self, name=None):
        super(GetOnes, self).__init__(name=name)

    def build(self, input_shape):
        super(GetOnes, self).build(input_shape)

    def call(self, tensor):
        output = tf.ones_like(tensor)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class AddValue(tf.keras.layers.Layer):
    """ Custom keras version of add operation. It adds a single value to a
        tensor. It's useful to shift the net output range. Since the keras
        doesn't allow a wrapper around layers, it's using tensorflow operation
        directly
    """
    def __init__(self, value, name=None):
        super(AddValue, self).__init__(name=name)
        self.value = tf.constant(value, tf.float32)
        self.function = None

    def build(self, input_shape):
        self.function = lambda tensor: tf.math.add(tensor, self.value)
        super(AddValue, self).build(input_shape)

    def call(self, tensor):
        output = self.function(tensor)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class DepthToSpace(tf.keras.layers.Layer):
    """ Keras layer implementation for depth to space operation of tensorflow
    """
    def __init__(self, block_size, data_format='NHWC', name=None):
        super(DepthToSpace, self).__init__(name=name)
        self.block_size = block_size
        self.data_format = data_format
        self.depth_to_space = None

    def build(self, input_shape):
        self.depth_to_space = lambda tensor: \
            tf.nn.depth_to_space(tensor, block_size=self.block_size,
                                 data_format=self.data_format)
        super(DepthToSpace, self).build(input_shape)

    def call(self, tensor):
        output = self.depth_to_space(tensor)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = np.array(input_shape.as_list())
        output_shape[1:3] *= self.block_size
        output_shape[3] /= (2 * self.block_size)
        return output_shape
