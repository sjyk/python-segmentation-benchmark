import numpy as np
np.seterr(divide='ignore') # these warnings are usually harmless for this code
from matplotlib import pyplot as plt
import copy, os
from generate.TrajectoryDataGenerator import *
from tsc.tsc import TransitionStateClustering


#creates a system whose regimes are uniformly sampled from the stochastic params
sys = createNewDemonstrationSystem(k=3,dims=2, observation=[0.0,0.1], resonance=[0.0,1], drift=[0,0.0])

#lm is the mean number of loops, dp is the probability of "missing"
#t = sampleDemonstrationFromSystem(sys,np.ones((2,1)), lm=0, dp=0)
#u = sampleDemonstrationFromSystem(sys,np.ones((2,1)), lm=0, dp=0)
#v = sampleDemonstrationFromSystem(sys,np.ones((2,1)), lm=0, dp=0)
#w = sampleDemonstrationFromSystem(sys,np.ones((2,1)), lm=0, dp=0)
#x = sampleDemonstrationFromSystem(sys,np.ones((2,1)), lm=0, dp=0)

a = TransitionStateClustering(window_size=2)
for i in range(0,20):
	t = sampleDemonstrationFromSystem(sys,np.ones((2,1)), lm=0, dp=0)
	a.addDemonstration(np.squeeze(t))

a.fit(normalize=False, pruning=0.9)
print a.segmentation

"""
SAVE_FIGURES = False

print('''
This demo shows the HDP-HSMM in action. Its iterations are slower than those for
the (Sticky-)HDP-HMM, but explicit duration modeling can be a big advantage for
conditioning the prior or for discovering structure in data.
''')

###############
#  load data  #
###############

T = 1000
data = np.loadtxt(os.path.join(os.path.dirname(__file__),'example-data.txt'))[:T]

#########################
#  posterior inference  #
#########################

# Set the weak limit truncation level
Nmax = 25

# and some hyperparameters
obs_dim = data.shape[1]
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
posteriormodel.add_data(data,trunc=60) # duration truncation speeds things up when it's possible

for idx in progprint_xrange(150):
    posteriormodel.resample_model()

posteriormodel.plot()

plt.show()
"""
