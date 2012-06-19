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

from User import *

class ExpertBase:
    def __init__(self):
        self.reset()
    
    # This class can be used as an iterator
    def __iter__(self):
        return self

    def reset(self):
        pass
    
    # This function accepts state information of the world (the nature outcome)
    def push(self,observation):
        pass

    # This function returns the next advice of the expert
    def next(self):
        return 0

    # This function updates internal data, based on the user's choice
    def update(self,info):
        pass

class ExpertConst(ExpertBase):
    advice = 0;
    def __init__(self,advice=0):
        ExpertBase.__init__(self)
        self.advice = advice

    def next(self):
        return self.advice

class ExpertCycle(ExpertBase):
    advice = 0;
    cycle_len = 1;
    def __init__(self,cycle_len=1):
        ExpertBase.__init__(self)
        self.advice = 0;
        self.cycle_len = cycle_len;

    def next(self):
        return self.advice

    def update(self,info):
        self.advice = (self.advice + 1) % self.cycle_len

class ExpertBest(ExpertBase):
    advice = 0;
    span = 4
    def __init__(self,span):
        ExpertBase.__init__(self)
        self.advice = 0;
        self.span = span

    def push(self,observation):
        self.advice=argmin(observation[:self.span])

    def next(self):
        return self.advice
   
class ExpertWorst(ExpertBase):
    advice = 0;
    span = 4
    def __init__(self,span):
        ExpertBase.__init__(self)
        self.advice = 0;
        self.span = span

    def push(self,observation):
        self.advice=argmax(observation[:self.span])

    def next(self):
        return self.advice
   
class ExpertSelfish(ExpertBase):
    advice = 0;
    span = 4
    def __init__(self,span):
        ExpertBase.__init__(self)
        self.advice = 0;
        self.span = span

    def push(self,observation):
        self.advice=argmin(observation[self.span:])

    def next(self):
        return self.advice
   
class ExpertSigma(ExpertBase):
    advice = 0;
    codex = dict()
    def __init__(self,policy,user):
        ExpertBase.__init__(self)
        self.advice = 0;
        self.codex = dict()
        if (policy.has_key('pi')):
            self.codex['pi']=policy['pi']
        else:
            raise KeyError
        if (policy.has_key('d_star')):
            self.codex['d_star']=policy['d_star']
        else:
            raise KeyError
        if (policy.has_key('sigma_p')):
            self.codex['sigma_p']=policy['sigma_p']
        else:
            raise KeyError
        if (policy.has_key('sigma')):
            self.codex['sigma']=policy['sigma']
            self.codex['inv_sigma']=scipy.linalg.inv(policy['sigma'])
        else:
            raise KeyError
        self.codex['user']=user

    def push(self,observation):
        ### Find nearest sigma point
        # Find differences between sigma points and the observation
        mat_tmp = self.codex['sigma_p'].T.copy()-observation
        # Calculate (squares) of difference norms
        norms = diag(dot(mat_tmp,dot(self.codex['inv_sigma'],mat_tmp.T)))
        #(mat_tmp*mat_tmp).sum(1)
        # Index of the nearest sigma point
        best_sigma = argmin(norms);

        (a_no,p_no) = self.codex['pi'].shape

        ### Calculate the action distribution at the best sigma point
        q = self.codex['d_star'][0:-1].copy()+\
            self.codex['d_star'][-1]*self.codex['pi'][:,best_sigma]

        ### Calculate the discrepancy of reproducable distributions
        # diffs = []
        # for a_idx in range(a_no):
        #     d = self.codex['user'].next_d([])
        #     q_a = d[0:-1].copy()
        #     q_a[a_idx]=q_a[a_idx]+d[-1]
        #     v = q-q_a;
        #     #diffs.append(dot(v,dot(self.codex['inv_sigma'],v)))
        #     diffs.append(scipy.linalg.norm(v))

        ## Compute the proportion of action that is missing from the
        ## current point when compared to the closest sigma point
        d = self.codex['user'].next_d([])
        q_a_cut = d[0:-1].copy()
        v = q-q_a_cut
        advice_distro = v.copy()
        for a_i in range(a_no):
            if (v[a_i]<0.0):
                advice_distro[a_i]=0.0
        if (advice_distro.sum()<=0):
            advice_distro = 1.0-v
        #advice_distro = exp(-1.0*array(diffs))
        advice_distro = advice_distro/advice_distro.sum()
        #self.advice = argmin(diffs)
        self.advice = argmax(random.multinomial(1,advice_distro))

    def next(self):
        return self.advice
