import pyhsmm
from pyhsmm.util.text import progprint_xrange
import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt

from pyhsmm.basic.distributions import NegativeBinomialIntegerR2Duration
import autoregressive.models as m
import autoregressive.distributions as di

"""
Uses an off the shelf HiddenSemiMarkovModel
"""
class HiddenSemiMarkovModel:

	def __init__(self, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		
		#internal variables not for outside reference
		self._demonstrations = []
		self._demonstration_sizes = []

	def addDemonstration(self,demonstration):
		demonstration = np.squeeze(np.array(demonstration))
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[Bayes] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)

		state_augmented = np.zeros((demo_size[0],2*demo_size[1]))
		state_augmented[:,0:demo_size[1]] = demonstration
		state_augmented[0:demo_size[0]-1,demo_size[1]:2*demo_size[1]] = demonstration[1:demo_size[0],:]
		state_augmented[demo_size[0]-1,demo_size[1]:2*demo_size[1]] = state_augmented[demo_size[0]-1,0:demo_size[1]]
		#state_augmented[:,2*demo_size[1]] = np.arange(0,demo_size[0],1)

		#state_augmented = preprocessing.normalize(state_augmented, axis=0)

		self._demonstrations.append(state_augmented)


	"""
	Essentially taken from Matt Johnson's demo
	"""
	def fit(self):
		data = np.squeeze(np.array(self._demonstrations[0])) #np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:T]
		Nmax = 25

		# and some hyperparameters
		obs_dim = data.shape[1]
		print data.shape
		obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.25,
                'nu_0':obs_dim+2}
		dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

		obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
		dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

		posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        	alpha=6.,gamma=6., # these can matter; see concentration-resampling.py
        	init_state_concentration=6., # pretty inconsequential
        	obs_distns=obs_distns,
        	dur_distns=dur_distns)
		
		for d in self._demonstrations:
			posteriormodel.add_data(np.squeeze(np.array(d)),trunc=60) # duration truncation speeds things up when it's possible

		for idx in progprint_xrange(50):
			posteriormodel.resample_model()

		new_segments = []
		for i in range(0, len(self._demonstrations)):
			new_segments.append(self.findTransitions(posteriormodel.states_list[i].stateseq))

		self.segmentation = new_segments
		self.model = posteriormodel

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
Uses an off the shelf AutoregressiveMarkovModel
"""
class AutoregressiveMarkovModel:
	def __init__(self, lag=4, alpha=1.5, gamma=4, nu=2, init_state_concentration=10, verbose=True):
		self.verbose = verbose
		self.segmentation = []
		self.model = []
		self.lag = lag
		self.alpha = alpha
		self.nu = nu
		self.gamma = gamma
		#self.cap=cap
		self.init_state_concentration = init_state_concentration
		
		#internal variables not for outside reference
		self._demonstrations = []
		self._demonstration_sizes = []

	def addDemonstration(self,demonstration):
		demonstration = np.squeeze(np.array(demonstration))
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[Bayes] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)
		self._demonstrations.append(demonstration)

	"""
	Essentially taken from Matt Johnson's demo
	"""
	def fit(self):
		p = self._demonstration_sizes[0][1]

		Nmax = self._demonstration_sizes[0][0]
		affine = True
		nlags = self.lag
		obs_distns=[di.AutoRegression(
    				nu_0=self.nu, S_0=np.eye(p), M_0=np.zeros((p,2*p+affine)),
    				K_0=np.eye(2*p+affine), affine=affine) for state in range(Nmax)]

		dur_distns=[NegativeBinomialIntegerR2Duration(
    				r_discrete_distn=np.ones(10.),alpha_0=1.,beta_0=1.) for state in range(Nmax)]

		model = m.ARWeakLimitHDPHSMMIntNegBin(
        alpha=self.alpha,gamma=self.gamma,init_state_concentration=self.init_state_concentration,
        	obs_distns=obs_distns,
        	dur_distns=dur_distns,
        )


		for d in self._demonstrations:
			model.add_data(d,trunc=60)

		#model.resample_model()

		for itr in progprint_xrange(20):
			model.resample_model()

		new_segments = []
		for i in range(0, len(self._demonstrations)):
			#print model.states_list[i].stateseq
			new_segments.append(self.findTransitions(model.states_list[i].stateseq))

		self.segmentation = new_segments
		self.model = model

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
