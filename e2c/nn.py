#!/bin/env python

'''
neural network helper functions
'''

import tensorflow as tf
import numpy as np
import ipdb as pdb
import re

def orthogonal_initializer(scale = 1.1):
  """
  From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
  """
  def _initializer(shape, dtype=tf.float32):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    print('Warning -- You have opted to use the orthogonal_initializer function')
    return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
  return _initializer

def linear(x,output_dim):
  w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=orthogonal_initializer(1.1))
  b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  return tf.matmul(x,w)+b

def ReLU(x, output_dim, scope):
  # helper function for implementing stacked ReLU layers
  with tf.variable_scope(scope):
    return tf.nn.relu(linear(x,output_dim))

def var_with_decay(name, shape, stddev, wd):
  # TODO - add weight decay
  var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
  return var

# CONVOLUTION-TYPE FUNCTIONS

def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def conv2d(x,ksize,name):
  # convolutional layers with pooling. no nonlinearity added
  with tf.variable_scope(name) as scope:
    in_channels=x.get_shape()[3].value
    assert(in_channels == ksize[2])
    kernel = tf.get_variable('weights',ksize, initializer=tf.truncated_normal_initializer(stddev=1e-3))
    conv=tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding='SAME')
    bias=tf.get_variable('biases',[ksize[3]],initializer=tf.constant_initializer(0.))
    conv=tf.nn.bias_add(conv,bias,name=name)
    activation_summary(conv)
    return conv

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')