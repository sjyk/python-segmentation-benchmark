import numpy as np
from sklearn import mixture, decomposition
from sklearn import hmm

"""
The approach from Calinon et al.
using a GMM with time as a feature
optimized using the bayesian information
criterion.

Same input and output behavior as TSC
"""
class TimeVaryingGaussianMixtureModel:

	def __init__(self, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		
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
	def fit(self, max_segments=20, hard_param=-1):

		if self.verbose:
			print "[Alternates] Clearing old model and segmentation"
		
		self.segmentation = []
		self.model = []


		new_segments = []
		new_model = []

		for d in self._demonstrations:
			gmm_list = []

			if hard_param == -1:
				for k in range(1,max_segments):
					g = mixture.GMM(n_components=k)
					g.fit(d) 
					gmm_list.append((g.bic(d),g)) #lower bic better
			else:
				g = mixture.GMM(n_components=hard_param)
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

	def __init__(self, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		
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
	def fit(self, n_components):

		if self.verbose:
			print "[Alternates] Clearing old model and segmentation"
		
		self.segmentation = []
		self.model = []


		new_segments = []
		new_model = []

		g = hmm.GaussianHMM(n_components=n_components)

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


