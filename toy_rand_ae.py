#!/bin/env python

'''
dead-simple VAE 
'''

import tensorflow as tf
import simulation.sim1D as sim # simulation interface
from policies.randpolicy import RandomPolicy # exploration policy
import ipdb as pdb
import numpy as np
import os

# DATA DIRECTORY
DATA_PATH='/ltmp/ae1d_simple'
ckpt_prefix="vae"

if not os.path.exists(DATA_PATH):
  os.makedirs(DATA_PATH)

x_dim=(1,)
u_dim=1
num_cycles=30 # total overall cycles
B=100 # num minibatches per cycle
batch_size=128

data_size = 1500

k=.2
A=int(k*data_size) # number of samples we gather on each cycle


policy_eval = RandomPolicy(1, x_dim, u_dim)
policy_batch = RandomPolicy(batch_size, x_dim, u_dim)

u=tf.placeholder(tf.float32,[batch_size,u_dim])

DATA_NAME = os.path.join(DATA_PATH,"D.npz")

def linear(x,output_dim):
  w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(0.,.01))
  b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  return tf.matmul(x,w)+b

# dead-simple AE
x=tf.placeholder(tf.float32,[batch_size,1])
u=tf.placeholder(tf.float32,[batch_size,1])
x_next=tf.placeholder(tf.float32,[batch_size,1])
with tf.variable_scope("hidden"):
  z=tf.tanh(linear(x,3))
with tf.variable_scope("hidden2"):
  z=tf.tanh(linear(z,3))
with tf.variable_scope("out"):
  x_recons=tf.sigmoid(linear(z,1))
with tf.variable_scope("predict"):
  h = tf.concat(1,[u,z])
  x_predict=tf.sigmoid(linear(h,1))

loss_recons = tf.square(x-x_recons) # data is 1D anyway
loss_predict = tf.square(x_next-x_predict)
loss = loss_predict + loss_recons
loss_scalar = tf.reduce_mean(loss) # total loss scalar

# easier to visualize for domain (0,1)
# loss_abs_r = tf.abs(x-x_recons)
# loss_abs_p = tf.abs(x_next-x_predict)

loss_log = tf.log(loss)

saver = tf.train.Saver(max_to_keep=num_cycles)
sess=tf.InteractiveSession()
optimizer=tf.train.AdamOptimizer(1e-3, beta1=0.1, beta2=0.1) # beta2=0.1
train_op=optimizer.minimize(loss_scalar)

def eval_1d(fetch, x0v,u0v,x1v):
  # evaluates an input tensor
  N=x0v.shape[0]
  L=np.zeros((N,1))
  for i in range(N // batch_size):
    s = i*batch_size
    e = (i+1)*batch_size
    feed_dict = {x:x0v[s:e,:], u:u0v[s:e,:], x_next:x1v[s:e,:]}
    L[s:e,:]=sess.run(fetch,feed_dict)
  feed_dict={x:x0v[-batch_size:,:], u:u0v[-batch_size:,:], x_next:x1v[-batch_size:,:]}
  L[-batch_size:,:]=sess.run(fetch,feed_dict)
  return L

def run_experiment():
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  optimizer=tf.train.AdamOptimizer(1e-3, beta1=0.1, beta2=0.1) # beta2=0.1
  train_op=optimizer.minimize(loss_scalar)
  summary = tf.scalar_summary("loss", loss_scalar)

  sess.run(tf.initialize_all_variables())

  # initial dataset
  D = np.zeros((data_size,3))
  x0=sim.init()
  # run Explore Policy over a long trajectory
  for i in range(data_size):
    u0=policy_eval.eval(sess, np.array(x0,ndmin=2))
    x1=sim.step(u0)
    D[i,:]=[x0,u0,x1]
    x0=x1

  # main training loop
  t=0
  for c in range(num_cycles):
    # 'Day' phase
    E = np.zeros((A,3))
    for i in range(A):
      u0=policy_eval.eval(sess,x0) # run Explore Policy
      x1=sim.step(u0)
      E[i,:]=[x0,u0,x1]
      x0=x1

    # replace dataset
    idx_d = np.random.choice(data_size,size=A,replace=False)
    D[idx_d,:] = E

    # update e2c
    for i in range(B):
      idx=np.random.randint(data_size,size=batch_size)
      x0v = D[idx,0].reshape((batch_size,1))
      u0v = D[idx,1].reshape((batch_size,1))
      x1v = D[idx,2].reshape((batch_size,1))
      e2c_res = sess.run([loss_scalar, train_op, summary],{x:x0v, u:u0v, x_next:x1v})
      writer.add_summary(e2c_res[2], t)
      t+=1
    print('cycle=%d e2c loss: %f' % (c, e2c_res[0]))
    
    # save trained result for current cycle along with samples used to train this iteration
    saver.save(sess, os.path.join(DATA_PATH,ckpt_prefix), global_step=c)
    np.savez(os.path.join(DATA_PATH, "data_%d.npz" % c), D=D, new=idx_d)


if __name__ == '__main__':
  run_experiment()
  print('done!')
