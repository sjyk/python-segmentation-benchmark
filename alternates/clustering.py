import numpy as np
from sklearn import mixture, decomposition
from sklearn import hmm
import coreset

"""
The approach from Calinon et al.
using a GMM with time as a feature
optimized using the bayesian information
criterion.

Same input and output behavior as TSC
"""
class TimeVaryingGaussianMixtureModel:

	def __init__(self, max_segments=20, hard_param=-1, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		self.max_segments = max_segments
		self.hard_param = hard_param
		
		#internal variables not for outside reference
		self._demonstrations = []
		self._demonstration_sizes = []

	def addDemonstration(self,demonstration):
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[Alternates] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)

		time_augmented = np.zeros((demo_size[0],demo_size[1]+1))
		time_augmented[:,0:demo_size[1]] = demonstration
		time_augmented[:,demo_size[1]] = np.arange(0,demo_size[0],1)

		self._demonstrations.append(time_augmented)

	#this fits using the BIC, unless hard param is specified
	def fit(self):

		if self.verbose:
			print "[Alternates] Clearing old model and segmentation"
		
		self.segmentation = []
		self.model = []


		new_segments = []
		new_model = []

		for d in self._demonstrations:
			gmm_list = []

			if self.hard_param == -1:
				for k in range(1,self.max_segments):
					g = mixture.GMM(n_components=k)
					g.fit(d) 
					gmm_list.append((g.bic(d),g)) #lower bic better
			else:
				g = mixture.GMM(n_components=self.hard_param)
				g.fit(d) 
				gmm_list.append((g.bic(d),g))


			gmm_list.sort()

			new_segments.append(self.findTransitions(gmm_list[0][1].predict(d)))
			new_model.append(gmm_list[0][1])

		self.segmentation = new_segments
		self.model = new_model

	#this finds the segment end points
	def findTransitions(self, predSeq):
		transitions = []
		prev = -1
		for i,v in enumerate(predSeq):
			if prev != v:
				transitions.append(i)
				#print i
			prev = v 

		transitions.append(i)
		return transitions

"""
The approach uses a HMM with Gaussian Emissions
"""
class HMMGaussianMixtureModel:

	def __init__(self, n_components, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		self.n_components = n_components
		
		#internal variables not for outside reference
		self._demonstrations = []
		self._demonstration_sizes = []

	def addDemonstration(self,demonstration):
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[Alternates] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)

		self._demonstrations.append(demonstration)

	#this fits using the BIC, unless hard param is specified
	def fit(self):

		if self.verbose:
			print "[Alternates] Clearing old model and segmentation"
		
		self.segmentation = []
		self.model = []


		new_segments = []
		new_model = []

		g = hmm.GaussianHMM(n_components=self.n_components)

		g.fit(self._demonstrations) 
			
		for d in self._demonstrations:
			new_segments.append(self.findTransitions(g.predict(d)))
			new_model.append(g)

		self.segmentation = new_segments
		self.model = new_model

	#this finds the segment end points
	def findTransitions(self, predSeq):
		transitions = []
		prev = -1
		for i,v in enumerate(predSeq):
			if prev != v:
				transitions.append(i)
				#print i
			prev = v 

		transitions.append(i)
		return transitions

"""
Coresets
"""
class CoresetSegmentation:
	def __init__(self, n_components, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		self.n_components = n_components
		
		#internal variables not for outside reference
		self._demonstrations = []
		self._demonstration_sizes = []

	def addDemonstration(self,demonstration):
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[Alternates] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)

		time_augmented = np.zeros((demo_size[0],demo_size[1]+1))
		time_augmented[:,0:demo_size[1]] = demonstration
		time_augmented[:,demo_size[1]] = np.arange(0,demo_size[0],1)

		self._demonstrations.append(time_augmented)

	#this fits using the BIC, unless hard param is specified
	def fit(self):

		if self.verbose:
			print "[Alternates] Clearing old model and segmentation"
		
		self.segmentation = []
		self.model = []


		new_segments = []
		new_model = []

		total_size = np.sum([ds[0] for ds in self._demonstration_sizes])
		data_matrix = np.zeros((total_size,self._demonstration_sizes[0][1]+1))
		i = 0
		for d in self._demonstrations:
			N = np.shape(d)
			data_matrix[i:i+N[0],:] = d
			i = i + N[0]

		new_model = coreset.get_coreset(data_matrix, self.n_components,self.n_components)[0]

		self.segmentation = self.taskToTrajectory(new_model)
		self.model = new_model

	def taskToTrajectory(self, new_model):
		result = []
		for d in self._demonstrations:
			Nm = np.shape(new_model)
			s = []
			for i in range(0,Nm[0]):
				l = []
				N = np.shape(d)
				for j in range(0,N[0]):
					l.append((np.linalg.norm(d[j,:]-new_model[i,:]),j))
				l.sort()
				s.append(l[0][1])
			s.append(0)
			s.append(N[0])
			result.append(s)
		return result



