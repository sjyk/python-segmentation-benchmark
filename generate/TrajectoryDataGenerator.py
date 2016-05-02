import numpy as np
import scipy
from scipy import linalg
from scipy.misc import comb
from SequenceDataGenerator import *

"""
This module provides the functions
to generate trajectory data of the 
specified dimensionality 
and characteristcs
"""

#generates a new system that we can sample demonstrations from
def createNewDemonstrationSystem(k=5, 
								 dims=1, 
								 bounds=[-10,10],
								 drift=[0,0], 
								 resonance=[0,0], 
								 observation=[0,0]):

	#np.random.seed(0)
	targets = generateTargetStates(k, dims, bounds)
	print targets
	system =  generateSystemSpecs(k,drift,resonance,observation) 
	
	return {'targets': targets,
			'system': system,
		    'bounds': bounds, 
		    'segments':k, 
		    'dimensionality':dims}

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(data, nTimes=50):
    points = [(i[0,0],i[1,0]) for i in data]
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    result = []
    for i in range(0,nTimes):
    	pt = np.zeros((2,1))
    	pt[0] = xvals[i]
    	pt[1] = yvals[i]
    	result.append(pt)

    return result

#from the start position runs the system til it reached end
#fits a bezier curve with the control points to avoid discretization
#effects
def interpolateToTarget(start, 
					    end,
					    specs,  
					    maxiter=100):
	cur = start
	traj = []
	i = 0

	#run until less than tol or number of iterations exceeded
	traj.append(start)

	while i < maxiter:
		traj.append(cur + np.random.randn(2,1)*specs[1])
		cur = cur + (np.transpose(end)-start)/maxiter
		i = i + 1

	traj.append(np.transpose(end)+ np.random.randn(2,1)*specs[0])

	interpolated = reversed(bezier_curve(traj))

	return [t + np.random.randn(2,1)*specs[2] for t in interpolated ]

#sample from the system
#lm is the amount of looping
#dp is the amount of dropping
def sampleDemonstrationFromSystem(sys,start,lm=0, dp=0):
	seq = generateRandomSequence(sys['segments'],lm,dp)
	traj = [start]
	prev = -1
	prevStart = start

	for j in seq:

		#code for looping, go back to the previous start
		if prev == j:
			s = traj[-1]
			newSeg = interpolateToTarget(s,
									     prevStart,
									     sys['system'][j])
			traj.extend(newSeg)
		else:
			prevStart = np.transpose(traj[-1])
		
		#normal execution
		s = traj[-1]
		newSeg = interpolateToTarget(s,
									 np.matrix(sys['targets'][j,:]),
									 sys['system'][j])
		traj.extend(newSeg)

		prev = j

	return (traj, seqToGroundTruth(seq))

"""
The ground truth time sequence
"""
def seqToGroundTruth(seq, nsteps=50):
	gt = [0]
	prev = -1
	for i in seq:
		if prev == i:
			gt.append(gt[-1]+2*nsteps)
		else:
			gt.append(gt[-1]+nsteps)

		prev = i

	return gt

###Plot sample
def plotData(traj):
	import matplotlib.pyplot as plt
	X = [t[0] for t in traj]
	Y = [t[1] for t in traj]
	plt.plot(X, Y, 'ro-')
	plt.show()

