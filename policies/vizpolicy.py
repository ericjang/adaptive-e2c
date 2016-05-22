#!/bin/env python

from policy import Policy

import tensorflow as tf
import numpy as np
from e2c import nn


class VisuomotorPolicy(Policy):
  '''
  primitive policy designed for simple 1D task
  adding a random sampling component allows the network to explore
  the space.
  '''
  def __init__(self, batch_size, x_dim, u_dim, name, share=None):
    super(VisuomotorPolicy, self).__init__(batch_size,x_dim,u_dim)
    self.name=name
    self.buildModel(share)

  def buildModel(self,share):
    with tf.variable_scope(self.name, reuse=share) as vs:
      self.x = tf.placeholder(tf.float32, (self.batch_size,) + self.x_dim, name="x") # image input
      c=self.x.get_shape()[3].value
      kernel=tf.get_variable('weights', [3,4,c,16], initializer=tf.truncated_normal_initializer(stddev=1e-4))
      conv=tf.nn.conv2d(self.x,kernel,[1,1,1,1],padding='SAME')
      biases=tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.))
      bias=tf.nn.bias_add(conv,biases)
      conv1=tf.nn.relu(bias,name=scope.name)
      # pool1
      pool1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
      # norm1
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

      #self.u=tf.tanh(sampleNormal(mu,tf.exp(logsigma)),name="u")
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

Policy.register(VisuomotorPolicy)
