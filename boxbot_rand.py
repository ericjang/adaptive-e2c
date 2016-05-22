#!/bin/env python

'''
TODO need to debug dimensions and construction of E2C model.

currently running out of memory when trying to run this model.

'''

import tensorflow as tf
from e2c.e2c_boxbot_model import E2CBoxbotModel
from simulation import boxbot_sim
import numpy as np
import ipdb as pdb
import os
import matplotlib.pyplot as plt
from policies.policy import Policy


# E2C Parameters
num_episodes=50 # total overall cycles
B=100 # num minibatches per cycle
batch_size=128
data_size = 500
k=.1
A=int(k*data_size) # number of samples we gather on each cycle

class RandomPolicy(Policy):
  def __init__(self, batch_size, x_dim, u_dim):
    super(RandomPolicy, self).__init__(batch_size, x_dim, u_dim)
  def eval(self, sess, x):
    return np.random.uniform(low=-5.,high=5.,size=self.u_dim)
    #np.random.randn(self.u_dim)
Policy.register(RandomPolicy)


DATA_PATH='/ltmp/e2c-boxbot-rand'


robot_type = "polyp" #"octoarm" # walker, polyp


def run_experiment():
  #tmp - verify E2C model builds properly
  x0v = np.zeros((120,320,6))
  u_dim=20
  u=tf.placeholder(tf.float32, [batch_size, u_dim])
  e2c = E2CBoxbotModel(x0v, u, batch_size)
  for v in tf.all_variables():
    print("%s : %s" % (v.name, v.get_shape()))
  sess=tf.InteractiveSession()
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  e2c.buildLoss(lambd=.25)
  e2c.buildTrain(learning_rate=1e-4)
  e2c.buildSummaries()
  pdb.set_trace()
  #end tmp

  if not os.path.exists(DATA_PATH):
      os.makedirs(DATA_PATH)
  ckpt_prefix="e2c"
  
  # if grpc not launched from own fn, it doesn't unblock?
  host = "gurney"#"localhost" #"gurney"
  # x_dim is the dim of 1 frame
  # x0v is two images
  (x0v, x_dim, u_dim) = boxbot_sim.init(robot_type, host=host, port=50051, draw=True) # start C++ sim
  policy_eval = RandomPolicy(1, x_dim, u_dim)

  x0v = x0v.astype(np.float32)/255

  # E2C TRAINING
  u=tf.placeholder(tf.float32, [batch_size, u_dim])
  e2c = E2CBoxbotModel(x0v, u, batch_size)
  for v in tf.all_variables():
    print("%s : %s" % (v.name, v.get_shape()))
  sess=tf.InteractiveSession()
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  e2c.buildLoss(lambd=.25)
  e2c.buildTrain(learning_rate=1e-4)
  e2c.buildSummaries()
  # re_init_p = tf.initialize_variables(policy_batch.policy_vars)
  # re_init_e = tf.initialize_variables(e2c.e2c_vars)
  sess.run(tf.initialize_all_variables())

  ## DATASET
  D={}
  D['x0'] = np.zeros((data_size,) + x_dim) # current + prev frame
  D['u0'] = np.zeros((data_size, u_dim))
  D['x1'] = np.zeros((data_size,) + x_dim) # current frame + next frame

  # pre-populate the dataset
  for i in range(data_size):
    u0v = policy_eval.eval(None,x0v) # (1,u_dim)
    x1v = boxbot_sim.step(u0v, draw=False).astype(np.float32)/255
    # store data
    D['x0'][i,...] = x0v
    D['u0'][i,:] = u0v.flatten()
    D['x1'][i,...] = x1v
    x0v = x1v
    print i
  # data verification
  # fig,axarr = plt.subplots(1,3)
  # axarr[0].imshow(x0v[:,:,:3])
  # axarr[1].imshow(x0v[:,:,3:])
  # axarr[2].imshow(x1v[:,:,3:])
  # plt.show()

  t=0 # train iterations
  l_hist = np.zeros(num_episodes*B)
  for c in range(num_episodes):
    Dp={}
    Dp['x0'] = np.zeros((A,) + x_dim) # current + prev frame
    Dp['u0'] = np.zeros((A, u_dim))
    Dp['x1'] = np.zeros((A,) + x_dim) # current frame + next frame

    # pre-populate the dataset
    for i in range(A):
      u0v = policy_eval.eval(None,x0v) # (1,u_dim)
      x1v = boxbot_sim.step(u0v, draw=False).astype(np.float32)/255
      Dp['x0'][i,...] = x0v
      Dp['u0'][i,:] = u0v.flatten()
      Dp['x1'][i,...] = x1v
      x0v = x1v

    # consolidate memories
    idx_new = np.random.choice(data_size,size=A,replace=False)
    for key in ['x0','u0','x1']:
      D[key][idx_new,...] = Dp[key]

    # train e2c
    for i in range(B):
      idx = np.random.randint(data_size,size=batch_size)
      x0v = D['x0'][idx,...]
      u0v = D['u0'][idx,:]
      x1v = D['x1'][idx,...]
      e2c_res = e2c.update(sess,(x0v,u0v,x1v),write_summary=True)
      writer.add_summary(e2c_res[2], t)
      l_hist[t]=e2c_res[0]
      t+=1

    saver.save(sess, os.path.join(DATA_PATH,ckpt_prefix), global_step=c)
    print('cycle=%d e2c loss: %f' % (c, e2c_res[0]))
  np.save(os.path.join(DATA_PATH,"l_hist"),l_hist)


if __name__ == '__main__':
  run_experiment()

