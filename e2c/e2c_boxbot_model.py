#!/bin/env python

'''
x can be arbitrary tensor shape
recons_loss is responsible for flattening it.
'''

from e2c_model import E2CModel
import tensorflow as tf
import nn
import ipdb as pdb



class E2CBoxbotModel(E2CModel):
  def __init__(self, x0, u, batch_size):
    z_dim=2
    u_dim = u.get_shape()[1].value
    super(E2CBoxbotModel, self).__init__(x0.shape, z_dim, u_dim, batch_size, u)

  def encode(self,x,share=None):
    with tf.variable_scope("encoder",reuse=share):
      # kernel shape = kernel shape is filter_height, filter_width, in_channels, out_channels
      conv1 = nn.conv2d(x,[5,5,6,64],'conv1')
      pool1 = nn.max_pool_2x2(tf.nn.relu(conv1))
      conv2 = nn.conv2d(pool1,[5,5,64,32],'conv2')
      pool2 = nn.max_pool_2x2(tf.nn.relu(conv2))
      conv3 = nn.conv2d(pool2,[5,5,32,32],'conv3')
      pool3 = nn.max_pool_2x2(tf.nn.relu(conv3))
      pool3 = tf.reshape(pool3,[self.batch_size,-1]) # reshape
      #pool3 = tf.reshape(pool1,[self.batch_size,-1])
      l1 = nn.ReLU(pool3,512,"fc1") # flatten and pass through ReLU
      h_enc = nn.ReLU(l1,512,"h_enc")
      return h_enc

  def dynamics(self, z):
    with tf.variable_scope("dynamics"):
      l1=nn.ReLU(z,100,"l1")
      h_trans = nn.ReLU(l1,100,"h_trans")
      return h_trans

  def decode(self, z, share=None):
    # reverse architecture to encode
    # here, h_dec should be same size as image
    H,W,C=self.x_dim
    with tf.variable_scope("decoder",reuse=share):
      l1 = nn.ReLU(z,512,"l1")
      l2 = nn.ReLU(l1,512,"l2")
      in_channels=512
      l2 = tf.reshape(l2,[self.batch_size,1,1,in_channels]) # single pixels with 512 channels
      kernel=tf.get_variable('kernel',[5,5,C,in_channels],initializer=tf.truncated_normal_initializer(stddev=1e-3))
      output_size=[self.batch_size,H,W,C]
      deconv1=tf.nn.deconv2d(l2,kernel,output_size,[1,1,1,1],padding='SAME')
      h_dec = tf.reshape(l2,[self.batch_size,-1])
      pdb.set_trace()
      return h_dec

  def sampleP_theta(self,h_dec,share=None):
    # override to avoid linear image * h_dec shape
    with tf.variable_scope("P_theta",reuse=share):
      x= tf.sigmoid(h_dec) # mean of bernoulli distribution
      return tf.reshape(x,(-1,)+self.x_dim) # reshape to original image dimensions
      
'''
class E2CBoxbotModel(E2CModel):
  def __init__(self, x,u,z_dim):
    super(E2CBoxbotModel, self).__init__(x, u, z_dim)

  def decode(self, z, share=None):
    with tf.variable_scope("decoder",reuse=share):
      for l in range(2):
        z=ReLU(z,200,"l"+str(l))
      # up-convolution
      with tf.variable_scope('deconv1') as scope:
        h,w,out_c=self.x_dim
        in_channels=200
        z=tf.reshape(z,(-1,1,1,in_channels))
        kernel=tf.get_variable('weights',[3,4,out_c,in_channels],initializer=tf.truncated_normal_initializer(stddev=1e-4))
        output_size=[self.batch_size,h,w,out_c]
        deconv1=tf.nn.deconv2d(z,kernel,output_size,[1,1,1,1],padding='SAME')
        return deconv1

  def sampleP_theta(self,h_dec,share=None):
    # override to avoid linear
    with tf.variable_scope("P_theta",reuse=share):
      x= tf.sigmoid(h_dec) # mean of bernoulli distribution
      return x
'''