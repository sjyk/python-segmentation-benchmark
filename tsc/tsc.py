"""
This class implements the basic transition state clustering
framework (TSC).


Transition State Clustering: Unsupervised Surgical 
Trajectory Segentation For Robot Learning. ISRR 2015. 

@inproceedings{krishnan2015tsc,
  title={Transition State Clustering: Unsupervised Surgical 
  Trajectory Segmentation For Robot Learning},
  author={Krishnan*, Sanjay and Garg*, Animesh and 
  Patil, Sachin and Lea, Colin and Hager, Gregory and 
  Abbeel, Pieter and Goldberg, Ken},
  booktitle={International Symposium of Robotics Research},
  year={2015},
  organization={Springer STAR}
}
"""
import numpy as np
from sklearn import mixture, decomposition
from dpcluster import *
import matplotlib.pyplot as plt

#debug
# import IPython

class TransitionStateClustering:

	"""
	A TransitionStateClustering model is constructed with
	a window size and initializes a bunch of internal state
	"""
	def __init__(self, window_size=2, verbose=True):
		self.window_size = window_size
		self.verbose = verbose
		self.model = []
		self.task_segmentation = []
		self.segmentation = []
		
		#internal variables not for external reference
		self._demonstrations = []
		self._transitions = []
		self._transition_states_scluster = []
		self._demonstration_sizes = []
		self._distinct_state_clusters = 0

	"""
	This function adds a demonstration to the model
	N x p (N time steps) (p dimensionality)
	"""
	def addDemonstration(self,demonstration):
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[TSC] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)
		self._demonstrations.append(demonstration)

	"""
	This function checks the sizes of the added demonstrations to make sure
	the dimensionality is consistent, and it returns None if inconsistent
	or the total length of all demonstration if consistent. 
	"""
	def checkSizes(self):
		if len(self._demonstration_sizes) == 0:
			return None

		first_element_p = self._demonstration_sizes[0][1]
		total_size =  self._demonstration_sizes[0][0]

		for i in range(1, len(self._demonstration_sizes)):
			if self._demonstration_sizes[i][1] != first_element_p:
				return None
			else:
				total_size = total_size  + self._demonstration_sizes[i][0]

		return total_size

	"""
	This function fits the TSC Model to demonstration data
	it takes a pruning threshold and some DP-GMM hyperparameters
	which are set to reasonable defaults
	"""
	def fit(self, 
			pruning=0.9,
			normalize=False,
			normalizeKern="rbf"):

		#first validate
		totalSize = self.checkSizes()

		if totalSize == None:
			raise ValueError("All of the demonstrations must have the same dimensionality")

		if self.verbose:
			print "[TSC] Clearing previously learned model"

		N = len(self._demonstration_sizes)
		self.model = []
		self.task_segmentation = []
		self.segmentation = []

		#helper routines
		self.identifyTransitions(totalSize,normalize,normalizeKern)
		self.clusterInState()
		self.pruneClusters(pruning)
		self.clusterInTime()
		self.taskToTrajectory()

	"""
	This prunes transitions to a specified threshold
	"""
	def pruneClusters(self,pruning):
		distinct_clusters = set([c[2] for c in self._transition_states_scluster])
		N = len(self._demonstration_sizes)
		new_transitions = []
		for c in distinct_clusters:
			tD = set([d[0] for d in self._transition_states_scluster if d[2] == c])
			tS = [d for d in self._transition_states_scluster if d[2] == c]
			if (len(tD) +0.0)/N > pruning:
				new_transitions.extend(tS)

		if self.verbose:
			print "[TSC] Transitions Before Pruning=", self._transition_states_scluster, "After=",new_transitions

		self._transition_states_scluster = new_transitions

	"""
	Takes the task segmentation and returns a trajectory
	segmentation. For conditioning reasons this doesn't 
	use DP-GMM but finds all clusters of size segmentl (automatically set)
	"""
	def taskToTrajectory(self):
		N = len(self._demonstration_sizes)
		for i in range(0,N):
			tSD = [(k[2],k[3],k[1]) for k in self.task_segmentation if k[0] == i]
			
			timeDict = {}
			for t in tSD:
				key = (t[0], t[1])
				if  key in timeDict:
					timeDict[key].append(t[2])
				else:
					timeDict[key] = [t[2]]
			
			tseg = [np.median(timeDict[k]) for k in timeDict]
			tseg.append(0)
			tseg.append(self._demonstration_sizes[i][0]-self.window_size)
			self.segmentation.append(tseg)


	"""
	This function identifies transition times in each demonstration
	"""
	def identifyTransitions(self, total_size, normalize, normalizeKern):
		p = self._demonstration_sizes[0][1]
		demo_data_array = np.zeros((total_size-self.window_size,p*self.window_size))

		inc = 0
		for i in range(0,len(self._demonstrations)):
			n = self._demonstration_sizes[i][0]
			for j in range(self.window_size,n):
				window = self._demonstrations[i][j-self.window_size:j,:]
				demo_data_array[inc,:] = np.reshape(window,(1,p*self.window_size))
				inc = inc + 1

		if self.verbose:
			print "[TSC] Created a window model with w=",self.window_size

		if normalize:
			kpca = decomposition.KernelPCA(p*self.window_size,kernel=normalizeKern)
			demo_data_array = kpca.fit_transform(demo_data_array)
			if self.verbose:
				print "[TSC] Normalizing With Kernel Transformation"

		"""
		Apply DP-GMM to find transitions
		"""
		indices = self.smoothing(self.DPGMM(demo_data_array, p*self.window_size))

		#print indices
		
		if self.verbose:
			print "[TSC] Removing all previously learned transitions"
		
		inc = 0
		self._transitions = []
		for i in range(0,len(self._demonstrations)):
			n = self._demonstration_sizes[i][0]
			for j in range(self.window_size,n):

				#by default the first/last state is a transition
				#otherwise it is the states where the indices are different
				if inc == 0 or j == self.window_size:
					pass#self._transitions.append((i,0))
				elif j == (n-1):
					pass#self._transitions.append((i,n-1))
				elif indices[inc-1] != indices[inc]:
					self._transitions.append((i,j-self.window_size))

				inc = inc + 1

		if self.verbose:
			print "[TSC] Discovered Transitions (demoid, time): ", self._transitions

	"""
	This applies smoothing to the indices to make sure
	rapid changes are discouraged
	"""
	def smoothing(self, indices):
		newIndices = indices
		for i in range(1,len(indices)):
			if indices[i] != indices[i-1] and indices[i] != indices[i+1] and indices[i+1] == indices[i-1]:
			   newIndices[i] = indices[i+1]

			   if self.verbose:
			   	print "[TSC] Smoothed out index=",i

		return newIndices


	"""
	Uses Teodor's code to do DP GMM clustering
	"""
	def DPGMM(self,data, dimensionality):
		vdp = VDP(GaussianNIW(dimensionality))
		vdp.batch_learn(vdp.distr.sufficient_stats(data))		
		likelihoods = vdp.pseudo_resp(np.ascontiguousarray(data))[0]

		real_clusters = 1
		cluster_s = vdp.cluster_sizes()
		total = np.sum(cluster_s)
		running_total = cluster_s[0]
		for i in range(1,len(vdp.cluster_sizes())):
			running_total = running_total + cluster_s[i]
			real_clusters = i + 1
			if running_total/total > 0.90:
				break

		return [np.argmax(l[0:real_clusters]) for l in likelihoods]

	"""
	This function applies the state clustering
	"""
	def clusterInState(self):
		tsN = len(self._transitions)
		p = self._demonstration_sizes[0][1]
		ts_data_array = np.zeros((tsN,p))

		for i in range(0, tsN):
			ts = self._transitions[i]
			ts_data_array[i,:] = self._demonstrations[ts[0]][ts[1],:]


		#Apply the DP-GMM to find the state clusters
		indices = self.DPGMM(ts_data_array,p)
		indicesDict = list(set(indices))

		self._transition_states_scluster = []
		self._distinct_state_clusters = 0
		
		if self.verbose:
			print "[TSC] Removing previously learned state clusters "

		#encode the first layer of clustering:
		for i in range(0,tsN):
			label = indicesDict.index(indices[i])
			tstuple = (self._transitions[i][0], self._transitions[i][1], label)
			self._transition_states_scluster.append(tstuple)

		self._distinct_state_clusters = len(list(set(indices)))
		#print self._distinct_state_clusters

		if self.verbose:
			print "[TSC] Discovered State Clusters (demoid, time, statecluster): ", self._transition_states_scluster

	"""
	This function applies the time sub-clustering
	"""
	def clusterInTime(self):
		p = self._demonstration_sizes[0][1]

		unorderedmodel = []

		for i in range(0,self._distinct_state_clusters):
			tsI = [s for s in self._transition_states_scluster if s[2]==i]
			ts_data_array = np.zeros((len(tsI),p))
			t_data_array = np.zeros((len(tsI),1))
			
			for j in range(0, len(tsI)):
				ts = tsI[j]
				ts_data_array[j,:] = self._demonstrations[ts[0]][ts[1],:]
				t_data_array[j,:] = ts[1]

			if len(tsI) == 0:
				continue

			#Since there is only one state-cluster use a GMM
			mm  = mixture.GMM(n_components=1)
			mm.fit(ts_data_array)

			#subcluster in time
			indices = self.DPGMM(t_data_array,1)
			indicesDict = list(set(indices))

			#finish off by storing two values the task segmentation	
			for j in range(0, len(tsI)):
				self.task_segmentation.append((tsI[j][0],
										  	   tsI[j][1],
										       tsI[j][2],
										       indicesDict.index(indices[j])))

			#GMM model
			unorderedmodel.append((np.median(t_data_array),mm))

		unorderedmodel.sort()
		self.model = [u[1] for u in unorderedmodel]

		if self.verbose:
			print "[TSC] Learned The Following Model: ", self.model
			






