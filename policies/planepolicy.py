#!/bin/env python

from policy import Policy

import tensorflow as tf
import numpy as np
from e2c import nn
import ipdb as pdb

def sampleNormal(mu,sigma):
  # diagonal stdev
  n01=tf.random_normal(sigma.get_shape(), mean=0, stddev=1) # use stdev .1 instead
  return mu+sigma*n01

class PlanePolicy(Policy):
  '''
  primitive policy designed for simple 1D task
  adding a random sampling component allows the network to explore
  the space.
  '''
  def __init__(self, batch_size, x_dim, u_dim, name, share=None):
    super(PlanePolicy, self).__init__(batch_size,x_dim,u_dim)
    self.name=name
    self.buildModel(share)

  def buildModel(self,share):
    with tf.variable_scope(self.name, reuse=share) as vs:
      self.x = tf.placeholder(tf.float32, (self.batch_size,np.prod(self.x_dim)), name="x") # flattened inptu
      l1=nn.ReLU(self.x,150,"l1")
      l2=nn.ReLU(l1,150,"l2")
      mu,logsigma=tf.split(1,2,nn.linear(l2,2*self.u_dim))
      self.u=tf.tanh(sampleNormal(mu,tf.exp(logsigma)),name="u")
    self.policy_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    print([v.name for v in self.policy_vars])

  def eval(self,sess,x):
    return sess.run(self.u,{self.x:x})
    
  def set_reward(self,r):
    # set objectie to minimize tensor -R
    self.reward = r # scalar
    self.buildTrain(1e-4)
    self.buildSummaries()

  def buildTrain(self,learning_rate):
    with tf.variable_scope("Optimizer"):
      optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.1, beta2=0.1) # beta2=0.1
      # maximize reward
      self.train_op=optimizer.minimize(-self.reward, var_list=self.policy_vars)

  def buildSummaries(self):
    tf.scalar_summary("R", self.reward)
    self.all_summaries = tf.merge_all_summaries()

  def update(self,sess,feed_dict, write_summary=False):
    fetches=[self.reward,self.train_op]
    if write_summary:
      fetches.append(self.all_summaries)
    return sess.run(fetches,feed_dict)

Policy.register(PlanePolicy)
