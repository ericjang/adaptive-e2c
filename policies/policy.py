#!/bin/env python

# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/models/image/cifar10/cifar10.py

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from e2c.nn import *
import ipdb as pdb

class Policy(object):
  __metaclass__ = ABCMeta
  """abstract policy \pi : X -> U that maps single observation
  directly to control"""
  def __init__(self, batch_size, x_dim, u_dim):
    super(Policy, self).__init__()
    self.x_dim=x_dim
    self.u_dim=u_dim
    self.batch_size=batch_size
  
  @abstractmethod
  def eval(self, x):
    return NotImplemented

  # def getOutput(self):
  #   return tf.placeholder(tf.float32, (self.batch_size,self.u_dim))
