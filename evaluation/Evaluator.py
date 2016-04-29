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

#on a list of systems it applies
#a list of algorithms
def run_k_system(system_list, 
					 algorithm_list,
					 initalcond,
					 metric, 
					 k=20,
					 lm=0, 
					 dp=0):

	result = np.zeros((len(system_list),len(algorithm_list)))
	for i,sys in enumerate(system_list):
		for j,a in enumerate(algorithm_list):
			result[i,j] = run_1_time(sys,a,initalcond,metric,k,lm,dp)[0]

	return result

#runs N trials of the same system specs
def run_comparison_experiment(params,
							  algorithm_list,
					 		  initalcond,
					 		  metric,
					 		  N=5, 
					 		  k=20,
					 		  lm=0, 
					 		  dp=0):
	system_list = []
	for i in range(0, N):
		system_list.append(createNewDemonstrationSystem(k=params['k'],
									   dims=params['dims'], 
									   observation=params['observation'], 
									   resonance=params['resonance'], 
									   drift=params['drift']))

	return run_k_system(system_list, 
						algorithm_list, 
						initalcond, 
						metric, 
						k=20,
					 	lm=0, 
					 	dp=0)

#can only sweep noise right now
def run_sweep_experiment(base_params,
						 sweep_param,
						 sweep_param_list,
						 algorithm_list,
						 initalcond,
					 	 metric,
					 	 N=5, 
					 	 k=20,
					 	 lm=0, 
					 	 dp=0):
	X = []
	Y = []

	for s in sweep_param_list:
		X.append(s)
		params = copy.deepcopy(base_params)
		params[sweep_param] = [params[sweep_param][0], s]
		result = run_comparison_experiment(params,
										  algorithm_list,
										  initalcond,
					 	 				  metric,
					 	 				  N, k,lm, dp)
		print np.mean(result,axis=0)
		Y.append(np.mean(result,axis=0))

	return (X,Y)

#plots the data structure that comes out of the parameter
#sweep
def plotY1Y2(points_tuple,
             title,
             xaxis,
             yaxis,
             legend=[],
             loc = 'upper right',
             filename="output.png",
             ylim=0,
             xlim=0):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	rcParams.update({'figure.autolayout': True})
	rcParams.update({'font.size': 18})
	fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf') 

	plt.figure() 
	colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
	shape = 's-'

	X = points_tuple[0]
	Y = points_tuple[1]
	num_algos = len(Y[0])

	for i in range(0, num_algos):
		ya = [j[i] for j in Y]
		plt.plot(X, ya, shape, linewidth=2.5,markersize=7,color=colors[i])

	plt.legend(legend,loc=loc)
	plt.title(title)
	plt.xlabel(xaxis,fontproperties=fprop)
	plt.ylabel(yaxis,fontproperties=fprop)
	plt.ylim(ymin=ylim) 
	plt.xlim(xmin=xlim, xmax=X[len(X)-1])
	plt.grid(True)
	plt.savefig(filename,bbox_inches='tight')
