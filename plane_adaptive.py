#!/bin/env python


import tensorflow as tf
from e2c.e2c_plane_model import E2CPlaneModel
from simulation.plane_sim import PlaneSim
import numpy as np
import ipdb as pdb
import os
import matplotlib.pyplot as plt
from policies.planepolicy import PlanePolicy


sim = PlaneSim('data/env2.png')
DATA_PATH='/ltmp/e2c-plane2-adaptive'
ckpt_prefix="e2c"

# num_episodes=50 # total overall cycles
# B=100 # num minibatches per cycle
# C=3
# extra_it = 50000
# batch_size=128
# data_size = 2000

num_episodes=500
B=100
C=3
batch_size=128
data_size=2000

x_dim=(40,40)
u_dim=2

k=.1
A=int(k*data_size) # number of samples we gather on each cycle

policy_eval=PlanePolicy(1, x_dim, u_dim, "epolicy")
policy_batch=PlanePolicy(batch_size, x_dim, u_dim, "epolicy", share=True)

e2c = E2CPlaneModel(policy_batch.u, batch_size)
for v in tf.all_variables():
  print("%s : %s" % (v.name, v.get_shape()))
e2c.buildLoss(lambd=.25)

policy_batch.set_reward(e2c.loss) # drive towards area where prediction is weak
sess=tf.InteractiveSession()

# save both policy and E2C variables
saver = tf.train.Saver(max_to_keep=num_episodes)

def getXs(D,idx):
  p0s = D[idx,0:2].reshape((batch_size,-1))
  u0v = D[idx,2:4]
  p1s = D[idx,4:6].reshape((batch_size,-1))

  x0v = np.zeros((batch_size,1600))
  x1v = np.zeros((batch_size,1600))
  for j in range(batch_size):
    x0v[j,:] = sim.getX(p0s[j,:]).flatten().astype(np.float32)
    x1v[j,:] = sim.getX(p1s[j,:]).flatten().astype(np.float32)
  return x0v,u0v,x1v

def run_experiment():
  if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
  # WRITER
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  e2c.buildTrain(learning_rate=1e-4)
  e2c.buildSummaries()
  re_init_p = tf.initialize_variables(policy_batch.policy_vars)
  re_init_e = tf.initialize_variables(e2c.e2c_vars)
  sess.run(tf.initialize_all_variables())

  # dataset 
  D = np.zeros((data_size,6)) # compressed format: Px,Py,Ux,Uy
  # initial data
  
  p0 = sim.init()
  for i in range(data_size):
    x0 = sim.getX(p0).reshape((1,-1)) # flatten
    u0 = policy_eval.eval(sess,x0)
    p1 = sim.step(u0)
    D[i,:]=np.concatenate([p0,u0.flatten(),p1])
    p0=p1

  E = np.zeros((A,6))
  t=0
  l_train = np.zeros(num_episodes*B)
  for c in range(num_episodes):
    for i in range(A):
      x0 = sim.getX(p0).reshape((1,-1)) # flatten
      u0 = policy_eval.eval(sess,x0)
      p1 = sim.step(u0)
      E[i,:]=np.concatenate([p0,u0.flatten(),p1])
      p0=p1

    # replace elements of dataset
    idx_new = np.random.choice(data_size,size=A,replace=False)
    D[idx_new,:] = E

    # train e2c
    for i in range(B):
      idx = np.random.randint(data_size,size=batch_size)
      x0v,u0v,x1v = getXs(D,idx)
      e2c_res = e2c.update(sess,(x0v,u0v,x1v),write_summary=False)
      #writer.add_summary(e2c_res[2], t)
      l_train[t]=e2c_res[0]
      t+=1
    print('cycle=%d e2c loss: %f' % (c, e2c_res[0]))

    sess.run(re_init_p) # this is crucial
    for i in range(C):
      for i in range(B):
        idx = np.random.randint(data_size,size=batch_size)
        x0v,u0v,x1v = getXs(D,idx)
        feed_dict = { policy_batch.x:x0v, e2c.x:x0v, e2c.x_next:x1v }
        p_res = policy_batch.update(sess, feed_dict, write_summary=False)
        #writer.add_summary(p_res[2], t)
        #l_hist[t]=p_res[0]
        #t+=1
      print('cycle=%d policy reward: %f' % (c, p_res[0]))
    
    # save trained data for this episode
    saver.save(sess, os.path.join(DATA_PATH,ckpt_prefix), global_step=c)
    np.savez(os.path.join(DATA_PATH, "data_%d.npz" % c), D=D, new=idx_new)

  # save the E2C loss / policy reward history
  np.save(os.path.join(DATA_PATH,"l_hist"),l_train)

  # post-train
  # sess.run(re_init_e) # should we do this to not get trapped in local min?
  # for i in range(extra_it):
  #   idx = np.random.randint(data_size,size=batch_size)
  #   x0v,u0v,x1v = getXs(D,idx)
  #   e2c_res = e2c.update(sess,(x0v,u0v,x1v),write_summary=False)
  #   if i % 1000 == 0:
  #     print('extra it=%d e2c loss: %f' % (i, e2c_res[0]))
  
if __name__ == '__main__':
  run_experiment()
  sess.close()
