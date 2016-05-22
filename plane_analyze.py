#!/bin/env python

"""
Generates figures for the plane task   
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import ipdb as pdb
from os.path import join

ADAPTIVE = True
if ADAPTIVE:
  import plane_adaptive as e
else:
  import plane_rand as e

def show_recons_samples(sess, episode):
  # visualize sample reconstructions
  e2c = e.e2c
  sim = e.sim
  ckptfile = join(e.DATA_PATH, "%s-%d" % (e.ckpt_prefix, episode))
  e.saver.restore(sess, ckptfile) # restore variable values
  tmp = np.load(join(e.DATA_PATH, "data_%d.npz" % episode))
  D = tmp['D']
  idx = np.random.randint(e.data_size,size=e.batch_size)
  x0v,u0v,x1v=e.getXs(D,idx)
  xr,xp=sess.run([e2c.x_recons, e2c.x_predict],feed_dict={e2c.x:x0v,e2c.u:u0v,e2c.x_next:x1v})
  A,B=e2c.A,e2c.B
  def getimgs(x,xnext):
    padsize=1
    padval=.5
    ph=B+2*padsize
    pw=A+2*padsize
    img=np.ones((10*ph,2*pw))*padval
    for i in range(10):
      startr=i*ph+padsize
      img[startr:startr+B,padsize:padsize+A]=x[i,:].reshape((A,B))
    for i in range(10):
      startr=i*ph+padsize
      img[startr:startr+B,pw+padsize:pw+padsize+A]=xnext[i,:].reshape((A,B))
    return img
  fig,arr=plt.subplots(1,2)
  arr[0].matshow(getimgs(x0v,x1v),cmap=plt.cm.gray,vmin=0,vmax=1)
  arr[0].set_title('Data')
  arr[1].matshow(getimgs(xr,xp),cmap=plt.cm.gray,vmin=0,vmax=1)
  arr[1].set_title('Reconstruction')
  plt.show()

def viz_z(sess, ckptfile):
  # does not actually use plane1.npz
  e2c = e.e2c
  e.saver.restore(sess,ckptfile) # restore variable values
  Ps,NPs=e.sim.getPSpace()
  batch_size=e2c.batch_size
  n0=NPs.shape[0]
  if False:
    Ps=np.vstack((Ps,NPs))
  xy=np.zeros([Ps.shape[0], 2])
  xy[:,0]=Ps[:,1]
  xy[:,1]=20-Ps[:,0] # for the purpose of computing theta, map centered @ origin
  Zs=np.zeros([Ps.shape[0], e2c.z_dim])
  theta=np.arctan(xy[:,1]/xy[:,0])
  for i in range(Ps.shape[0] // batch_size):
    print("batch %d" % i)
    x_val=e.sim.getXs(Ps[i*batch_size:(i+1)*batch_size,:])
    Zs[i*batch_size:(i+1)*batch_size,:]=sess.run(e2c.z, {e2c.x:x_val})
  # last remaining points may not fit precisely into 1 minibatch.
  x_val=e.sim.getXs(Ps[-batch_size:,:])
  Zs[-batch_size:,:]=sess.run(e2c.z, {e2c.x:x_val})

  if False:
    theta[-n0:]=1

  fig,arr=plt.subplots(1,2)
  arr[0].scatter(Ps[:,1], 40-Ps[:,0], c=(np.pi+theta)/(2*np.pi))
  arr[0].set_title('True State Space')
  arr[1].scatter(Zs[:,0],Zs[:,1], c=(np.pi+theta)/(2*np.pi))
  arr[1].set_title('Latent Space Z')
  #plt.show()
  return fig
 
def viz_z_unfold(sess, cpktprefix):
  #d=1000 # save interval
  #for i in range(int(1-00) // d):
  for i in range(1,100):
    #f="%s-%05d" % (cpktprefix,i*d)
    f="%s-%d" % (cpktprefix,i)
    print(f)
    fig=viz_z(sess,f)
    fig.suptitle('%d'%i)
    fig.savefig("e2c-%02d.png" % (i))
    # combine with convert -delay 10 -loop 0 e2c-plane-*.png out.gif
    # then reduce size using gifsicle -O3-colors 256 < out.gif > new.gif
  print('done!')

def loss_surf(ckptfile):
  '''
  computes expected loss of each state, over the distribution 
  of actions U (121 of them)
  128 * 1600 points.
  '''
  e.saver.restore(e.sess, ckptfile)
  e2c=e.e2c
  Ps,NPs=e.sim.getPSpace()
  L_e2c=np.zeros(Ps.shape[0])
  # deal with proper batching later
  u0v = np.zeros((e.batch_size,2))
  tmp0,tmp1 = np.meshgrid(np.linspace(-1,1,11), np.linspace(-1,1,11))
  u0v[:121,0] = tmp0.flatten()
  u0v[:121,1] = tmp1.flatten()
  for j in range(Ps.shape[0]):
    p0 = Ps[j,:] # r,c
    x0 = e.sim.getX(p0).reshape((1,-1))
    x0v = np.repeat(x0,e.batch_size,axis=0)
    # get predictions
    p1v = np.zeros((e.batch_size,2))
    for k in range(e.batch_size):
      p1v[k,:]=e.sim.fstep(p0,u0v[k,:])
    x1v = e.sim.getXs(p1v)
    # evaluate loss scalar (mean)
    res = e.sess.run(e2c.loss_vec,{e2c.x:x0v,e2c.u:u0v,e2c.x_next:x1v})
    L_e2c[j] = np.mean(res[:121])
  return L_e2c

def viz_tableau():
  # visualize where bot wandered in the dataset.
  e2c=e.e2c
  # cycles=[i*20 for i in range(4)]
  # cycles.append(99)
  Ps,NPs=e.sim.getPSpace()
  cycles = [i*(e.num_episodes // 5) for i in range(5)]
  cycles.append(e.num_episodes-1)
  num_plots = len(cycles)
  fig,axarr = plt.subplots(num_plots,2)
  # deal with proper batching later
  u0v = np.zeros((e.batch_size,2))
  tmp0,tmp1 = np.meshgrid(np.linspace(-1,1,11), np.linspace(-1,1,11))
  u0v[:121,0] = tmp0.flatten()
  u0v[:121,1] = tmp1.flatten()
  for i in range(num_plots):
    c=cycles[i]
    ckptfile= join(e.DATA_PATH, "%s-%d" % (e.ckpt_prefix, c))
    e.saver.restore(e.sess, ckptfile)
    tmp = np.load(join(e.DATA_PATH, "data_%d.npz" % c))
    D = tmp['D']
    idx_new=tmp['new']
    # column 0 - dataset distribution
    #axarr[i,0].hexbin(D[:,0],D[:,1], cmap=plt.cm.YlOrRd_r, gridsize=40, vmin=0)
    idx_old=[k for k in range(e.data_size) if k not in idx_new]
    # scatter X=c, Y=40-r 
    axarr[i,0].scatter(D[idx_old,1],40-D[idx_old,0],c='b')
    axarr[i,0].scatter(D[idx_new,1],40-D[idx_new,0],c='g') # scatter new points
    axarr[i,0].set(adjustable='box-forced',aspect='equal')
    axarr[i,0].set_xlim([0,40])
    axarr[i,0].set_ylim([0,40])
    # column 1 - expected loss over X0
    # i.e. marginal distribution of L(X)
    L_img = np.zeros((40,40))
    L = loss_surf(ckptfile)
    for j in range(Ps.shape[0]):
      p0 = Ps[j,:] # r,c
      L_img[int(p0[0]),int(p0[1])] = L[j]
    axarr[i,1].matshow(L_img,cmap=plt.cm.gray)
    print(c)
  # add y axis labels
  for i in range(num_plots):
    c=cycles[i]
    axarr[i,0].set_ylabel('cycle '+str(c))

def plot_losses():
  '''
  l_hist_r is num_episodes*B long
  while l_hist_a is num_episodes*(B+C*B) long
  '''
  prefix = '/data/people/evjang/capstone_data'
  l_hist_r = np.load(join(prefix,'l_hist_rand.npy'))
  l_hist_a = np.load(join(prefix,'l_hist_adaptive.npy'))
  l_hist_a2 = np.zeros(l_hist_r.size)
  B,C = e.B, e.C
  s=100 # start index (initial losses are usually huge)
  T = range(e.num_episodes // B)
  for c in range(e.num_episodes):
    l_hist_a2[c*B:c*B+B] = l_hist_a[c*(B+C*B):c*(B+C*B)+B]
  pdb.set_trace()
  plt.plot(T[s:],l_hist_r[s:],label='Random Policy')
  plt.plot(T[s:],l_hist_a2[s:],label='Adaptive Policy')
  plt.legend()

def compute_val_loss():
  '''
  compare E2C validation loss between simultaneous exploration / learning
  vs. learning after exploration
  '''
  fname = join(e.DATA_PATH, "%s-val-loss.npz" % (e.ckpt_prefix))
  L_mean = np.zeros(e.num_episodes)
  L_std = np.zeros(e.num_episodes)
  #l_b = np.zeros(e.num_episodes)
  for c in range(e.num_episodes):
    ckptfile =  join(e.DATA_PATH, "%s-%d" % (e.ckpt_prefix, c))
    L = loss_surf(ckptfile)
    L_mean[c] = np.mean(L)
    L_std[c] = np.std(L)
    print(c)
  T = range(e.num_episodes)
  np.savez(fname, mean=L_mean, std=L_std)
  plt.errorbar(T, L_mean, L_std)
  #plt.plot(T,l_a,label='Post-exploration Learning')
  #plt.plot(T,l_b,label='Simultaneous Learning')
  #plt.legend()


def plot_entropy():
  '''
  visualizes entropy of state distribution
  '''
  pass


if __name__=="__main__":
  #viz_z_unfold(sess, join(e.DATA_PATH,e.ckpt_prefix))
  ckpt_file = join(e.DATA_PATH, "%s-%d" % (e.ckpt_prefix, e.num_episodes-1))
  #plot_losses()
  fig=viz_z(e.sess,ckpt_file)
  viz_tableau()
  show_recons_samples(e.sess,e.num_episodes-1)
  plt.show()
  compute_val_loss()
  #show_recons_seq(sess, "/ltmp/e2c-plane-199000.ckpt")
  #plt.show()
  e.sess.close()