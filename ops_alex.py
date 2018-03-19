import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    assigners = []
    shadow_variables = []

    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, is_train, convolutional=True, decay=0.99, epsilon=1e-5, scale_after_normalization=True,
                 name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.convolutional = convolutional
            self.is_train = is_train
            self.epsilon = epsilon
            self.ema = tf.train.ExponentialMovingAverage(decay=decay)
            self.scale_after_normalization = scale_after_normalization
            self.name=name

    def __call__(self, x):
        shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            depth = shape[-1]
            self.gamma = tf.get_variable("gamma", shape=[depth],
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", shape=[depth],
                                initializer=tf.constant_initializer(0.))
            self.mean = tf.get_variable('mean', shape=[depth],
                                        initializer=tf.constant_initializer(0),
                                        trainable=False)
            self.variance = tf.get_variable('variance', shape=[depth],
                                        initializer=tf.constant_initializer(1),
                                        trainable=False)
            
            # Add to assigners if not already added previously.
            if not tf.get_variable_scope().reuse:
                batch_norm.assigners.append(self.ema.apply([self.mean, self.variance]))
                batch_norm.shadow_variables += [self.ema.average(self.mean), self.ema.average(self.variance)]

            if self.convolutional:
                x_unflattened = x
            else:
                x_unflattened = tf.reshape(x, [-1, 1, 1, depth])

            if self.is_train:
                if self.convolutional:
                    mean, variance = tf.nn.moments(x, [0, 1, 2])
                else:
                    mean, variance = tf.nn.moments(x, [0])

                assign_mean = self.mean.assign(mean)
                assign_variance = self.variance.assign(variance)
                with tf.control_dependencies([assign_mean, assign_variance]):
                    normed = tf.nn.batch_norm_with_global_normalization(
                        x_unflattened, mean, variance, self.beta, self.gamma, self.epsilon,
                        scale_after_normalization=self.scale_after_normalization)
            else:
                mean = self.ema.average(self.mean)
                variance = self.ema.average(self.variance)
                local_beta = tf.identity(self.beta)
                local_gamma = tf.identity(self.gamma)
                normed = tf.nn.batch_norm_with_global_normalization(
                      x_unflattened, mean, variance, local_beta, local_gamma,
                      self.epsilon, self.scale_after_normalization)
            if self.convolutional:
                return normed
            else:
                return tf.reshape(normed, [-1, depth])

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(axis=3, values=[x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.01,padding='SAME',
           name="conv2d",reuse=None):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim],
                            initializer=tf.constant_initializer(0.01))
        # if not tf.get_variable_scope().reuse:
        #     tf.summary.histogram(w.name, w)
        conv = tf.nn.bias_add(tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding),b)
        return conv

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,padding='SAME',
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # if not tf.get_variable_scope().reuse:
        #     tf.summary.histogram(w.name, w)
        return tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1],padding=padding)

def upconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,padding='SAME',
             name="upconv2d"):
    
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        new_h = input_.get_shape().as_list()[1]*d_h**2
        new_w = input_.get_shape().as_list()[2]*d_w**2
        upsized = tf.image.resize_images(input_, [new_h, new_w], method=1)

        w = tf.get_variable('w', [k_h, k_h,input_.get_shape()[-1], output_shape[-1] ],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # if not tf.get_variable_scope().reuse:
        #     tf.summary.histogram(w.name, w)
        return tf.nn.conv2d(upsized, w,strides=[1, d_h, d_w, 1],padding=padding)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope='Linear', stddev=0.02):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size],
                            initializer=tf.constant_initializer(0.02))
        # if not tf.get_variable_scope().reuse:
        #     tf.histogram_summary(matrix.name, matrix)
        return tf.matmul(input_, matrix) + b


def normalize_batch_of_images(batch_of_images):
    mean, var = tf.nn.moments(batch_of_images, [1,2], keep_dims=True)
    std = tf.sqrt(var)
    normed = (batch_of_images - mean) / std
    return normed

def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon))) 


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def conv(x, num_filters, filter_height, filter_width, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act