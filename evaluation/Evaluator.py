"""
The evaluator module runs multiple runs of 
an algorithm on different instances of the
same task.
"""
import copy
import numpy as np
from generate.TrajectoryDataGenerator import *

#system is a demonstration system
#algorithm is 
#metric a function seq x seq -> R
#K is the number of demonstrations
#also takes in an algorithm parameters
def run_1_time(system, 
				algorithm,
				initalcond,
				metric, 
				k=20,
				lm=0, 
				dp=0):

	a = copy.deepcopy(algorithm)
		
	gtlist = []
	for j in range(0,k):
		t = sampleDemonstrationFromSystem(system,initalcond,lm=lm, dp=dp)
		a.addDemonstration(np.squeeze(t[0]))
		gtlist.append(t[1])

	a.fit()
	result = []
	for j in range(0,k):
		print gtlist[j],a.segmentation[j]
		result.append(metric(gtlist[j],a.segmentation[j]))

	return (np.mean(result), np.var(result))


	