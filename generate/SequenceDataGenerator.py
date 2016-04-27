import numpy as np
import scipy
from scipy import linalg

"""
This module provides some primitives for
generating random sequences of data.
"""

#Loops are drawn from a poisson distribution
#with specified mean (lm) elements are dropped
#wp dp
def generateRandomSequence(k=5, lm=0, dp=0):
	seq = []
	for i in range(0,k):
		if np.random.rand(1,1) > dp:
			loops = np.random.poisson(lam=lm) + 1
			for j in range(0, loops):
				seq.append(i)
	return seq

#Generate targets for the specified state-space
def generateTargetStates(k=5, dims=1, bounds=[-10,10]):
	return np.random.rand(k,dims)*(bounds[1]-bounds[0])+bounds[0]*np.ones((k,dims))

#Generates a dynamical system with the specified noise properties
#selects uniformly at random from the specs
def generateSystemSpecs(k = 5,
						drift=[0,0], 
						resonance=[0,0], 
						observation=[0,0]):
	specs = []
	for i in range(0,k):
		a = np.random.rand()*drift[1] + drift[0]
		b = np.random.rand()*resonance[1] + resonance[0]
		c = np.random.rand()*observation[1] + observation[0]
		specs.append((a,b,c))
	return specs


