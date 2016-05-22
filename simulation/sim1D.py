#!/bin/env python

'''
Simple 1D environment
x bounded between 0,1 (image space)
u bounded by policy between -1,1
'''

import numpy as np
import matplotlib.pyplot as plt

u_dim=1
_x=None # state

def fn_V(x):
  y = x#(x+.5)*.5
  #return 0
  return np.minimum(500.*np.square((y)*(y-.6)*(y-1.)*(y+.4)),1)

def init():
  global _x
  _x = np.random.rand()
  return _x

def fstep(x,u):
  v = u*.1 # maximum step size in either direction
  return np.maximum(0,np.minimum(1, x + v*(1.-fn_V(x))))

def step(u):
  # x moves in direction of u, slowed down by potential
  # u is (-1,1)
  # we expect that over random samples u, agent spends most of
  # its time in areas of higher potentials (moves slowest)
  global _x
  _x = fstep(_x, u)
  return _x

def scatter_traj(ax, X,U):
  ax.scatter(X,U)
  ax.set_xlim([0,1])
  ax.set_ylim([-1, 1]) # u limits

def plot_traj(ax,X):
  ax.plot(X)
  ax.set_ylim([0,1]) 

def plot_V(ax):
  x=np.linspace(-1,1,100)
  ax.plot(x,fn_V(x))
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])

if __name__ == '__main__':
  # this demonstrates that a random (-1,1) policy has poor mixing because 
  # samples tend to be drawn from 
  T=1500
  x=init()
  X1=np.zeros(T)
  U0=(np.random.rand(T)-.5)*2. # (-1,1)
  for i in range(T):
    x = step(U0[i])
    X1[i]=x
    
  fig,(ax0,ax1,ax2)=plt.subplots(1,3)
  plot_V(ax0)
  scatter_traj(ax1,X1[:T-1],U0[1:T])
  plot_traj(ax2,X1)
  plt.show()


  