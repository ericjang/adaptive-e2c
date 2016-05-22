#!/bin/env python


from e2c_model import E2CModel
import tensorflow as tf
import nn
import numpy as np

class E2C1DModel(E2CModel):
  def __init__(self, u, batch_size):
    x_dim=(1,)
    z_dim=1
    u_dim=1
    super(E2C1DModel, self).__init__(x_dim, z_dim, u_dim, batch_size, u)
    
  def encode(self,x,share=None):
    # X -> H_ENC
    with tf.variable_scope("encoder",reuse=share):
      for l in range(3):
        x=nn.ReLU(x,5,"l"+str(l))
      return nn.linear(x,2*self.z_dim)

  def dynamics(self, z):
    # Z -> H_TRANS
    with tf.variable_scope("dynamics"):
      for l in range(3):
        z=nn.ReLU(z,5,"l"+str(l))
      return z

  def decode(self, z, share=None):
    # Z -> H_DEC
    with tf.variable_scope("decoder",reuse=share):
      for l in range(3):
        z=nn.ReLU(z,10,"l"+str(l))
      return z

if __name__ == '__main__':
  # run basic 1-D continuous plane task
  DATA_PATH='/ltmp/e2c1d_test'
  batch_size=128
  u = tf.placeholder(tf.float32, (batch_size,1))
  e2c = E2C1DModel(u, batch_size)
  e2c.buildLoss(lambd=.25)
  for v in tf.all_variables():
      print("%s : %s" % (v.name, v.get_shape()))
  e2c.buildTrain(learning_rate=1e-4)
  e2c.buildSummaries()
  sess=tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())
  writer = tf.train.SummaryWriter(DATA_PATH, sess.graph_def)
  for i in range(30000):
    x0v = np.random.uniform(0.,1.,batch_size).reshape((batch_size,1))
    uv = np.random.uniform(-.1,.1,batch_size).reshape((batch_size,1))
    x1v = np.minimum(0.,np.maximum(x0v + uv,1.))
    e2c_res = e2c.update(sess, (x0v,uv,x1v), write_summary=True)
    writer.add_summary(e2c_res[2], i)
    if i%100==0:
      print('i=%d e2c loss: %f' % (i, e2c_res[0]))
