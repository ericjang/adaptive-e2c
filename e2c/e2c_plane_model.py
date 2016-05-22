#!/bin/env python

from e2c_model import E2CModel
#import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import nn
import ipdb as pdb

class E2CPlaneModel(E2CModel):
  def __init__(self, u, batch_size):
    self.A=40
    self.B=40
    x_dim=(self.A*self.B,)
    z_dim=2
    u_dim=2
    super(E2CPlaneModel, self).__init__(x_dim, z_dim, u_dim, batch_size, u)

  def encode(self,x,share=None):
    with tf.variable_scope("encoder",reuse=share):
      l1=nn.ReLU(x,150,"l1")
      l2=nn.ReLU(l1,150,"l2")
      h_enc=nn.linear(l2,2*self.z_dim)
      return h_enc

  def dynamics(self, z):
    with tf.variable_scope("dynamics"):
      l1=nn.ReLU(z,100,"l1")
      h_trans = nn.ReLU(l1,100,"h_trans")
      return h_trans

  def decode(self, z, share=None):
    with tf.variable_scope("decoder",reuse=share):
      l1=nn.ReLU(z,200,"l1")
      l2=nn.ReLU(l1,200,"l2")
      h_dec=nn.linear(l2,self.x_dim_flat)
      return h_dec
  