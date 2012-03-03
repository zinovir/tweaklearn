# Load full scientific calculations support
from numpy import *
import scipy as Sci
import scipy.linalg
# Ensure you can save Matlab formated trace datafiles
# from scipy.io import mio
#import random

# Ensure you can work with files and system services
import sys
# Ensure basic math is available
import math

import Expert
import User

##### This app plays the role of the central plexus of a MAB. It holds
## and operates the state, its visibility and utility functions. It
## also controls the connectivity of the experts and the user.

class MABAppBase (dict):
    # Iteration counter
    count = 0;
    # outcome = [];

    # This function uses 'config' to extract the set of experts, the
    # user, and the list of utility functions. Default utilities are
    # available. The 'config' also contains parameters for the class's
    # outcome generation.
    def __init__(self,config):
        dict.__init__(self)
        if (config.has_key('experts')):
            self['experts']=config['experts']
        else:
            self['experts']=[]

        #print self['experts']

        if (config.has_key('user')):
            self['user']=config['user']
        else:
            raise KeyError
        self.reset()

    # The following function resets the simulation, if and when necessary
    def reset(self):
        self.count = 0;
        self['user'].reset()
        for x in self['experts']:
            x.reset()

    # The class is also an iterator
    def __iter__(self):
        return self

    # This function generates the system outcome
    def generate(self):
        return []

    # This function produces expert view of the outcome
    def expert_observe(self, outcome):
        return []

    # This is the default expert post-choice information function
    def expert_observe_choice(self,outcome,choice):
        return []

    # This is the default uesr post-choice information function
    def user_observe_choice(self,outcome,choice,expert_poll=[]):
        return []

    # This function produces the next complete iteration of the MAB
    # It 1) generates the outcome and its expert observation
    # It 2) informs the experts of their observation
    # It 3) polls the experts
    # It 4) informs the user of expert poll 
    # It 5) polls the user for choice
    # It 6) calculates choice dependent observations
    # It 7) informs all parties of their choice dependent observations
    # It 8) returns a complete (timed) trace of this step
    def next(self):
        # 1) Generate the outcome
        outcome = self.generate();
        # 1) Generate observation for the experts
        exp_obs = self.expert_observe(outcome);

        # 2) Inform the experts of what they should see
        for x in self['experts']:
            x.push(exp_obs)

        # 3) Poll the experts
        expert_poll = [x.next() for x in self['experts'] ]
        
        # 4) inform the user of expert poll
        self['user'].push(expert_poll)

        # 5) poll the user for choice
        choice = self['user'].next()

        # 6) calculate choice dependent observations
        #exp_info = [self.expert_observe_choice(outcome,choice) \
        #            for x in self['experts'] ]
        exp_info = self.expert_observe_choice(outcome,choice)
        user_info = self.user_observe_choice(outcome,choice,expert_poll)
        # 7) informs all parties of their choice dependent observations
        for x in self['experts']:
            x.update(exp_info)
        self['user'].update(user_info)
        
        # It 8) returns a complete (timed) trace of this step
        trace = [self.count,outcome,expert_poll,choice,exp_info,user_info]
        self.count = self.count +1
        return trace

# This class implements the transparent MAB with gaussian state
# The choice of the user is the choice of _expert_
# The MAB is transparent because the user receives utility info on all
# experts with advice equivalent to that of the chosen expert
class MABGaussTransparent (MABAppBase):
    gen_params = dict()
    a_coeff = 1.0
    b_coeff = 0.0
    n_choice = 4
    
    def __init__(self,config):
        MABAppBase.__init__(self,config)
        # Extract here the parameters of the Gauss distro, make sure
        # they are there
        if (config.has_key('gen_param')):
            self.gen_params = config['gen_param']
            if (not self.gen_params.has_key('mu')):
                raise KeyError
            if (not self.gen_params.has_key('sigma')):
                raise KeyError
            if (not self.gen_params.has_key('discr')):
                self.gen_params['discr']=4;
            if (config.has_key('n_choice')):
                self.n_choice = config['n_choice']
            else:
                self.n_choice = 4
            # Calculate observation correctors
            sigma = self.gen_params['sigma'];
            mu = self.gen_params['mu']
            discr = self.gen_params['discr']
            x_min = mu[0]-discr*sigma[0,0]
            x_max = mu[0]+discr*sigma[0,0]
            #print "x",x_min, x_max,mu[0]
            for idx in range(self.n_choice):
                val_min = mu[idx]-discr*sqrt(sigma[idx,idx])
                val_max = mu[idx]+discr*sqrt(sigma[idx,idx])
                #print "val",val_min, val_max
                if (x_min > val_min):
                    x_min = val_min
                if (x_max < val_max):
                    x_max = val_max
            self.a_coeff = 1.0/(x_max-x_min)
            self.b_coeff = -x_min*self.a_coeff
        else:
            raise KeyError
        #print self['experts']

    # This function generates the system outcome
    def generate(self):
        condition = True
        sample = []
        while (condition):
            sample = random.multivariate_normal(self.gen_params['mu'],\
                                                self.gen_params['sigma'])
            cor_sample = self.a_coeff*sample+self.b_coeff
            if ((min(cor_sample[0:self.n_choice])<0.0) or \
                 (max(cor_sample[0:self.n_choice])>1.0)):
                condition = True
            else:
                condition = False
        return sample

    # This function produces expert view of the outcome
    def expert_observe(self, outcome):
        return outcome

    # This is the default expert post-choice information function
    def expert_observe_choice(self,outcome,choice):
        return [outcome, choice]

    # This is the default uesr post-choice information function
    def user_observe_choice(self,outcome,choice,expert_poll):
        # The observation will comprise the utility, and then a list
        # of indices of those expert who gave the advice equivalent to
        # the chosen expert

        # Find chosen expert advice
        advice = expert_poll[choice]
        #print advice

        # Find all experts with equivalent advice
        idxs = []
        for idx in range(len(expert_poll)):
            if (expert_poll[idx]==advice):
                idxs.append(idx)

        #print outcome
        # Find utility -- notice it is mapped into [0,1]
        utility = self.a_coeff*outcome[advice]+self.b_coeff

        #return [utility,[choice]]
        return [utility,idxs]

