#!/bin/env python


import tensorflow as tf
import simulation.sim1D as sim # simulation interface
from policies.simplepolicy import SimplePolicy # exploration policy
import ipdb as pdb
import numpy as np
import os

# DATA DIRECTORY
DATA_PATH='/ltmp/ae1d_adaptive'
if not os.path.exists(DATA_PATH):
  os.makedirs(DATA_PATH)
ckpt_prefix="ae"

x_dim=(1,)
u_dim=1
num_cycles=30 # total overall cycles

B=100 # num minibatches per cycle
C=10 
k=.2

data_size=1500 # how big we want learning dataset to be

A=int(k*data_size) # number of samples we gather on each cycle


batch_size=128

stochastic_policy=True

policy_eval=SimplePolicy(1, x_dim, u_dim, "epolicy", stochastic=stochastic_policy)
policy_batch=SimplePolicy(batch_size, x_dim, u_dim, "epolicy", share=True, stochastic=stochastic_policy) # share parameters

def linear(x,output_dim):
  w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(0.,.01))
  b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  return tf.matmul(x,w)+b

# dead-simple AE for the E2C model
with tf.variable_scope('ae') as vs:
  x=tf.placeholder(tf.float32,[batch_size,1])
  u = policy_batch.u
  #u=tf.placeholder(tf.float32,[batch_size,1])
  x_next=tf.placeholder(tf.float32,[batch_size,1])
  with tf.variable_scope("hidden"):
    z=tf.tanh(linear(x,3))
  with tf.variable_scope("out"):
    x_recons=tf.sigmoid(linear(z,1))
  with tf.variable_scope("predict"):
    h = tf.concat(1,[u,z])
    x_predict=tf.sigmoid(linear(h,1))

e2c_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

loss_recons = tf.square(x-x_recons) # data is 1D anyway
loss_predict = tf.square(x_next-x_predict)
loss = loss_predict + loss_recons
loss_scalar = tf.reduce_mean(loss) # total loss scalar

# easier to visualize for domain (0,1)
loss_abs_r = tf.abs(x-x_recons)
loss_abs_p = tf.abs(x_next-x_predict)

policy_batch.set_reward(loss_scalar) # drive towards area where prediction is weak

saver = tf.train.Saver(max_to_keep=num_cycles)
sess=tf.InteractiveSession()

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

### MAIN EXPERIMENT ### 

def run_experiment():
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  optimizer=tf.train.AdamOptimizer(1e-3, beta1=0.1, beta2=0.1) # beta2=0.1
  train_op=optimizer.minimize(loss_scalar, var_list=e2c_vars)
  summary = tf.scalar_summary("loss", loss_scalar)
  # optimizer=tf.train.AdamOptimizer(1e-3, beta1=0.1, beta2=0.1)
  # train_policy=optimizer.minimize(-loss_scalar, var_list=policy_eval)
  re_init = tf.initialize_variables(policy_batch.policy_vars)

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
    
    # update policy
    sess.run(re_init) # re-initializing the exploration policy prevents us from getting stuck & saturating parameters
    for j in range(C): # epochs within episode
      for i in range(B): # num minibatches per epoch
        idx=np.random.randint(data_size,size=batch_size)
        x0v = D[idx,0].reshape((batch_size,1))
        u0v = D[idx,1].reshape((batch_size,1))
        x1v = D[idx,2].reshape((batch_size,1))
        feed_dict = { policy_batch.x:x0v, x:x0v, x_next:x1v }
        p_res = policy_batch.update(sess, feed_dict, write_summary=True)
        writer.add_summary(p_res[2], t)
        t+=1
      print('cycle=%d policy reward: %f' % (c, p_res[0]))
    
    # save trained result for current cycle along with samples used to train this iteration
    saver.save(sess, os.path.join(DATA_PATH,ckpt_prefix), global_step=c)
    np.savez(os.path.join(DATA_PATH, "data_%d.npz" % c), D=D, new=idx_d)

if __name__ == '__main__':
  run_experiment()
  print('done!')
