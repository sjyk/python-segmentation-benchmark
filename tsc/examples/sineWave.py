#!/usr/bin/env python

"""
This implements a sineWave example
"""
import os,sys,inspect
import numpy as np
from sklearn import mixture
from sklearn.externals.six.moves import xrange
import matplotlib.pyplot as plt
import matplotlib as mpl

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from tsc import *

# Number of samples per component
n_samples = 100

# Generate random sample following a sine curve
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4 * np.pi / n_samples

for i in xrange(X.shape[0]):
	x = i * step - 6
	X[i, 0] = x + np.random.normal(0, 0.1)
	X[i, 1] = 3 * (np.sin(x) + np.random.normal(0, .1))


# Number of samples per component
n_samples = 100

# Generate random sample following a sine curve
np.random.seed(0)
Y = np.zeros((n_samples, 2))
step = 4 * np.pi / n_samples

for i in xrange(Y.shape[0]):
	x = i * step - 6
	Y[i, 0] = x + np.random.normal(0, 0.1)
	Y[i, 1] = 1.0 * (np.sin(x) + np.random.normal(0, .1))

		
a = TransitionStateClustering(window_size=2)
a.addDemonstration(X)
a.addDemonstration(Y)
a.fit(normalize=True)

markers =['o','x','o','x','o','x','o']

plt.subplot(1,2,1)
a.segmentation[0].sort()

inc = 0
previ = 0
for i in a.segmentation[0]:
	plt.scatter(X[previ:i,0], X[previ:i,1], color='r', marker=markers[inc],s=50)
	previ = i
	print previ
	inc = inc + 1

inc = 0
previ = 0
a.segmentation[1].sort()
for i in a.segmentation[1]:
	plt.scatter(Y[previ:i,0], Y[previ:i,1], color='b', marker=markers[inc],s=50)
	previ = i
	inc = inc + 1



#plt.scatter(X[a.segmentation[0],0], X[a.segmentation[0],1], s=100,color='k')
#plt.scatter(Y[a.segmentation[1],0], Y[a.segmentation[1],1], s=100,color='k')

plt.xlim(-8, 4 * np.pi - 6+2)
plt.ylim(-5, 5)
plt.title("TSC With RBF Normalization")

a = TransitionStateClustering(window_size=2)
a.addDemonstration(X)
a.addDemonstration(Y)
a.fit(normalize=False)

plt.subplot(1,2,2)
inc = 0
previ = 0
a.segmentation[0].sort()
for i in a.segmentation[0]:
	plt.scatter(X[previ:i,0], X[previ:i,1], color='r', marker=markers[inc],s=50)
	previ = i
	print previ
	inc = inc + 1

inc = 0
previ = 0
a.segmentation[1].sort()
for i in a.segmentation[1]:
	plt.scatter(Y[previ:i,0], Y[previ:i,1], color='b', marker=markers[inc],s=50)
	previ = i
	inc = inc + 1

plt.xlim(-8, 4 * np.pi - 6+2)
plt.ylim(-5, 5)
plt.title("TSC Without RBF Normalization")
plt.show()