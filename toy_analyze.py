#!/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import numpy as np
import simulation.sim1D as sim # simulation interface
import ipdb as pdb
import os

# adaptive experiment or simple experiment
ADAPTIVE = False

if ADAPTIVE:
  import adaptive_ae as e
else:
  import simple_ae as e


cmap_g = sns.cubehelix_palette(rot=-.4, as_cmap=True, reverse=True)
cmap_b = sns.cubehelix_palette(start=2.8, rot=.1, as_cmap=True)
cmap_p = sns.cubehelix_palette(light=.85, as_cmap=True)

def tshist(ax,X,Y):
  B=25
  assert(X.size==Y.shape[1])
  img=np.zeros((B,X.size)) 
  for t in range(X.size):
    hc,be=np.histogram(Y[:,t],bins=B,range=(-1,1))
    img[::-1,t]=hc.astype(np.float)/Y.shape[0]
  sns.heatmap(img,ax=ax)

cycles=[i*6 for i in range(5)]
num_plots=len(cycles)
# generate the tableau
f, axarr = plt.subplots(num_plots, 4)
for i in range(num_plots): # each row
  c = cycles[i]
  e.saver.restore(e.sess, join(e.DATA_PATH, "%s-%d" % (e.ckpt_prefix, c)))
  
  # column 0 - dataset used to train this cycle
  data_name = os.path.join(e.DATA_PATH, "data_%d.npz" % c)
  tmp = np.load(data_name)
  D = tmp['D']
  if ADAPTIVE:
    idx_new = tmp['new']
    idx_old = [r for r in range(e.data_size) if r not in idx_new]
    axarr[i,0].scatter(D[idx_old,0],D[idx_old,1],c='b')
    axarr[i,0].scatter(D[idx_new,0],D[idx_new,1],c='g')
  else:
    axarr[i,0].scatter(D[:,0],D[:,1],c='b')

  # column 1 - conditional distributions P(U|X) after learning
  X=np.linspace(0,1,e.batch_size).reshape((e.batch_size,1))
  U=np.zeros((100,e.batch_size)) # samples, x histogram
  for s in range(100):
    U[s,:]=e.policy_batch.eval(e.sess,X).flat
  tshist(axarr[i,1],X,U)
  
  # column 2 loss surfaces after learning
  x0v,u0v=np.meshgrid(np.linspace(0,1,30), np.linspace(-1,1,30))
  x0v,u0v=x0v.reshape((-1,1)),u0v.reshape((-1,1))
  x1v = np.array([sim.fstep(x, u0v[j]) for j,x in enumerate(x0v)]).reshape((-1,1))
  Lr = e.eval_1d(e.loss_abs_r,x0v,u0v,x1v)
  sns.heatmap(Lr.reshape(30,30),ax=axarr[i,2],cmap=cmap_g,vmin=0.,vmax=1.)

  # column 3 - prediction loss under a variety of 
  Lp = e.eval_1d(e.loss_abs_p,x0v,u0v,x1v)
  sns.heatmap(Lp.reshape(30,30),ax=axarr[i,3],cmap=cmap_g,vmin=0.,vmax=1.)
  #Xr = e.eval_1d(e.loss_predict,X,X,X) # the second 2 args are hacks
  #axarr[i,3].plot(X,Xr)
  print(c)


# # test
# x0v = X
# u0v = np.ones((e.batch_size,1))
# x1v = np.ones((e.batch_size,1)) # irrelevant
# xp = e.eval_1d(e.x_predict,x0v,u0v,x1v)
# pdb.set_trace()

# label cycle numbers
for i in range(num_plots):
  axarr[i,0].set_ylabel("cycle " + str(cycles[i]))

# ugly
for i in range(num_plots): # rows
  for j in [0]:
    axarr[i,j].set_xlim((-.1,1.1)) # x
    if j==3:
      yl=(0.,1.)
    else:
      yl=(-1.1,1.1)
    axarr[i,j].set_ylim(yl)

axarr[0,0].set_title("Dataset")
axarr[0,1].set_title("Learned P(U|X)")
axarr[0,2].set_title("Reconstruction Loss")
axarr[0,3].set_title("Prediction Loss")

#plt.savefig("adaptiveAE.png")
e.sess.close()
plt.show()
