#!/bin/env python

from policy import Policy

import numpy as np

class RandomPolicy(Policy):
  '''
  random policy - for benchmarking E2C model 
  on a 'static' dataset
  '''
  def __init__(self, batch_size, x_dim, u_dim):
    super(RandomPolicy, self).__init__(batch_size, x_dim, u_dim)

  def eval(self, sess, x):
    return np.random.uniform(low=-1.,high=1.,size=[x.shape[0],self.u_dim])
    #np.random.randn(self.u_dim)

Policy.register(RandomPolicy)
