#!/bin/env python

'''
RPC interface to run boxbot simulator in C++
'''

import boxbot_pb2
from borg import xterm_cmd
from grpc.beta import implementations
import ipdb as pdb

import numpy as np

import matplotlib.pyplot as plt

_TIMEOUT_SECONDS = 1000
_stub=None
_udim=None
_x0 = None # previous _x0 is stored

def init(robot_type, host="localhost",port=50051, draw=False):
	'''
	initializes C++ process, then 
	sends RPC request to C++ server to initialize the simulation
	we get back x_dim, u_dim
	'''
	#xterm_cmd("/home/evjang/boxbot/boxbot server")
	# connect to it
	global _stub
	global _x0
	E = boxbot_pb2.ExperimentDef()
	E.robot = robot_type
	channel=implementations.insecure_channel(host,port)
	_stub=boxbot_pb2.beta_create_RPCSim_stub(channel)
	sp=_stub.init(E, _TIMEOUT_SECONDS)

	_udim=sp.u_dim

	x0=np.fromstring(sp.x.data, dtype=np.uint8)
	x0=np.flipud(x0.reshape((sp.x.height,sp.x.width,sp.x.channels)))

	_x0 = x0

	if draw:
		plt.imshow(x0)
		plt.show()
	x_dim=(sp.x.height, sp.x.width, 2*sp.x.channels) # two images stacked on top of each other
	x = np.concatenate([_x0, x0],axis=2)
	return (x, x_dim, sp.u_dim)

def step(u,draw=False):
	global _stub
	global _x0
	cdata=boxbot_pb2.ControlData()
	cdata.control_data.extend(u) # u needs to be flattened
	#odata=_stub.step(cdata, _TIMEOUT_SECONDS)
	o=_stub.step(cdata,_TIMEOUT_SECONDS)
	assert(len(o.data)==o.width*o.height*o.channels)
	x1=np.fromstring(o.data, dtype=np.uint8)
	#np.save('x.npy',x)
	x1=x1.reshape((o.height,o.width,o.channels))
	x1=np.flipud(x1)
	if draw:
		plt.imshow(x1)
		plt.show()
	x = np.concatenate([_x0, x1], axis=2)
	_x0 = x1
	return x
