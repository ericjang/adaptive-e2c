#!/bin/env python


import tensorflow as tf
from e2c.e2c_plane_model import E2CPlaneModel
from simulation.plane_sim import PlaneSim
import numpy as np
import ipdb as pdb
import os
import matplotlib.pyplot as plt
from policies.randpolicy import RandomPolicy

num_episodes=500 # total overall cycles
B=100 # num minibatches per cycle
batch_size=128
data_size = 2000
x_dim=(40,40)
u_dim=2

k=.1
A=int(k*data_size) # number of samples we gather on each cycle
sim = PlaneSim('data/env2.png')

u=tf.placeholder(tf.float32, [batch_size, u_dim]) # control at time t
e2c = E2CPlaneModel(u, batch_size)
for v in tf.all_variables():
  print("%s : %s" % (v.name, v.get_shape()))
e2c.buildLoss(lambd=.25)
policy_eval = RandomPolicy(1, x_dim, u_dim)

sess=tf.InteractiveSession()

DATA_PATH='/ltmp/e2c-plane2-rand'
ckpt_prefix="e2c"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

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

def experiment1():
  '''
  train E2C simultaneously with exploration
  '''
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  e2c.buildTrain(learning_rate=1e-4)
  e2c.buildSummaries()
  
  sess.run(tf.initialize_all_variables())

  # dataset 
  D = np.zeros((data_size,6)) # compressed format: Px,Py,Ux,Uy
  # initial data
  p0 = sim.init()
  for i in range(data_size):
    x0 = sim.getX(p0)
    u0 = policy_eval.eval(sess,x0)
    p1 = sim.step(u0)
    D[i,:]=np.concatenate([p0,u0.flatten(),p1])
    p0=p1

  t=0
  l_hist = np.zeros(num_episodes*B)
  for c in range(num_episodes):
    E = np.zeros((A,6))
    for i in range(A):
      x0 = sim.getX(p0)
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
      x0v,u0v,x1v=getXs(D,idx)
      e2c_res = e2c.update(sess,(x0v,u0v,x1v),write_summary=True)
      writer.add_summary(e2c_res[2], t)
      l_hist[t]=e2c_res[0]
      t+=1
    # save trained data for this iteration
    saver.save(sess, os.path.join(DATA_PATH,ckpt_prefix), global_step=c)
    np.savez(os.path.join(DATA_PATH, "data_%d.npz" % c), D=D, new=idx_new)
    print('cycle=%d e2c loss: %f' % (c, e2c_res[0]))
  np.save(os.path.join(DATA_PATH,"l_hist"),l_hist)
  
def experiment2():
  '''
  gather data first, then train
  '''
  e2c.buildTrain(learning_rate=1e-4)
  e2c.buildSummaries()
  sess.run(tf.initialize_all_variables())
  # explore first, then train
  D = np.zeros((data_size,6)) # compressed format: Px,Py,Ux,Uy
  # initial data
  p0 = sim.init()
  for i in range(data_size):
    x0 = sim.getX(p0)
    u0 = policy_eval.eval(sess,x0)
    p1 = sim.step(u0)
    D[i,:]=np.concatenate([p0,u0.flatten(),p1])
    p0=p1

  for c in range(num_episodes):
    E = np.zeros((A,6))
    for i in range(A):
      x0 = sim.getX(p0)
      u0 = policy_eval.eval(sess,x0)
      p1 = sim.step(u0)
      E[i,:]=np.concatenate([p0,u0.flatten(),p1])
      p0=p1

    # replace elements of dataset
    idx_new = np.random.choice(data_size,size=A,replace=False)
    D[idx_new,:] = E

  l_hist = np.zeros(num_episodes * B)
  t=0
  for c in range(num_episodes):
    for i in range(B):
      idx = np.random.randint(data_size,size=batch_size)
      x0v,u0v,x1v = getXs(D,idx)
      e2c_res = e2c.update(sess,(x0v,u0v,x1v),write_summary=False)
      l_hist[t]=e2c_res[0]
      t+=1
    print('cycle=%d e2c loss: %f' % (c, e2c_res[0]))
    saver.save(sess, os.path.join(DATA_PATH,ckpt_prefix), global_step=c)
    np.savez(os.path.join(DATA_PATH, "data_%d.npz" % c), D=D, new=idx_new)
  np.save(os.path.join(DATA_PATH,"l_hist"),l_hist)
  np.savez(os.path.join(DATA_PATH, "data_%d.npz" % c), D=D, new=idx_new)

if __name__ == '__main__':
  #experiment1()
  experiment2()
  sess.close()