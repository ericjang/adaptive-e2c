
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import randint
import os
import ipdb as pdb

class PlaneSim(object):
  def __init__(self, env_file):
    super(PlaneSim, self).__init__()
    self.im = plt.imread(env_file) # grayscale
    self.p = None
    self.w,self.h = 40,40
    self.rw = 1
  
  def init(self):
    self.p=np.array([self.rw,randint(self.rw,self.w-self.rw)])
    return self.p
  
  def getX(self,p):
    # return image X given true state p (position) of robot
    x=np.copy(self.im)
    x[p[0]-self.rw:p[0]+self.rw+1, p[1]-self.rw:p[1]+self.rw+1]=1. # robot is white on black background
    return x.reshape((1,self.h,self.w))

  def getXs(self, Ps):
    X=np.zeros((Ps.shape[0],1600))
    for i in range(Ps.shape[0]):
      X[i,:]=self.getX(Ps[i,:]).flatten()
    return X
  
  def is_colliding(self,p):
    if np.any([p-self.rw<0, p+self.rw>=self.w]):
      return True
    # check robot body overlap with obstacle field
    return np.mean(self.im[p[0]-self.rw:p[0]+self.rw+1, p[1]-self.rw:p[1]+self.rw+1]) > 0.05

  def step(self,u):
    self.p = self.fstep(self.p,u)
    return self.p

  def fstep(self,p0,u):
    p=np.copy(p0)
    d=np.sign(u.flat[0])
    nsteps=int(round(np.abs(u.flat[0])))
    for i in range(nsteps):
      p[0]+=d
      if self.is_colliding(p):
        p[0]-=d
        break
    d=np.sign(u.flat[1])
    nsteps=int(round(np.abs(u.flat[1])))
    for i in range(nsteps):
      p[1]+=d
      if self.is_colliding(p):
        p[1]-=d
        break
    return p

  def getPSpace(self):
    P=np.zeros((self.w*self.h,2)) # max possible positions
    NP=np.zeros((self.w*self.h,2))
    i,j=(0,0)
    p=np.array([self.rw,self.rw]) # initial location
    for dr in range(self.h):
      for dc in range(self.w):
        pp=p+np.array([dr,dc])
        if not self.is_colliding(pp):
          P[i,:]=pp
          i+=1
        else:
          NP[j,:]=pp
          j+=1
    return P[:i,:], NP[:j,:]

if __name__ == '__main__':
  # make sure it is doing what we think
  x=init("data/env1.png")
  max_dist=3
  T=10
  for i in range(T):
    u=randint(-max_dist,max_dist+1,size=2)
    step(u)
    plt.imshow(getX(p).reshape(h,w),cmap=plt.cm.gray)
    plt.show()