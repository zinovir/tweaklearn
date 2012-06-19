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

class UserBase:
    choice = 0

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

    # This function returns the next choice of the expert
    def next(self):
        return 0

    # This function return the next choice distribution of the expert
    def next_d(self,state):
        return array([0])

    # This function updates internal data, based on the user's choice
    def update(self,info):
        pass

class UserConst(UserBase):
    choice = 0;
    def __init__(self,choice=0):
        UserBase.__init__(self)
        self.choice = choice

    def next(self):
        return self.choice

class UserCycle(UserBase):
    choice = 0;
    cycle_len = 1;
    def __init__(self,cycle_len=1):
        UserBase.__init__(self)
        self.choice = 0;
        self.cycle_len = cycle_len;

    def next(self):
        return self.choice

    def update(self,info):
        #print "Old choice",self.choice,"\n"
        self.choice = (self.choice + 1) % self.cycle_len
        #print "New choice",self.choice,"and the cycle is",self.cycle_len,"\n"

class UserExp3(UserBase):
    counter = 0
    w = []
    p_expert = []
    n_choice = 0
    n_experts = 0
    beta = 1.0
    beta_spec = 0.75
    gamma = 0.1
    #advice = []
    
    def __init__(self,config):
        UserBase.__init__(self)
        if(config.has_key('n_choice')):
            self.n_choice = config['n_choice']
        else:
            raise KeyError
        if(config.has_key('n_experts')):
            self.n_experts = config['n_experts']
        else:
            raise KeyError
        if(config.has_key('beta')):
            self.beta = config['beta']
        else:
            self.beta = 1.0
        if(config.has_key('beta_spec')):
            self.beta_spec = config['beta_spec']
        else:
            self.beta_spec = 0.75
        if(config.has_key('gamma')):
            self.gamma = config['gamma']
        else:
            self.gamma = 0.1
        #self.w = [1.0]*self.n_experts
        
    def reset(self):
        print "User reset called"
        self.w = [1.0]*self.n_experts
        self.counter = 0

    
    # This function accepts state information of the world (the nature outcome)
    #def push(self,observation):
        # Record (but don't use) expert actions
        #self.advice = observation
        #pass

    # This function returns the next choice distribution
    def next_d(self,state):
        if(len(state)==0):
            state = array(self.w)
        # Compute the weighted distribution of experts
        a_tmp = array(state)
        p_expert = (1.0-self.gamma)*a_tmp/a_tmp.sum()+\
                        self.gamma/self.n_experts
        return p_expert

    # This function returns the next choice of the expert
    def next(self):
        # Compute the weighted distribution of experts
        self.p_expert = self.next_d(self.w)
        # Draw an expert due to above distro
        choice = argmax(random.multinomial(1,self.p_expert))
        # Return the expert -- this is the choice that the user makes
        return choice

    # This function updates internal data, based on the user's choice
    def update(self,info):
        # Update all weights based on the info (value of all mentioned experts)
        val,experts = info
        g = array(self.n_experts*[self.beta_spec])/self.p_expert
#        for exp_idx in experts:
#            self.w[exp_idx]=self.w[exp_idx]*\
#                             exp(-self.beta*val/self.p_expert[exp_idx])
        for exp_idx in experts:
            g[exp_idx] = g[exp_idx]+(1-val)/self.p_expert[exp_idx]

        self.w=self.w*exp(self.beta*g)
        #print self.w
        if (self.counter == 0):
            print "Weights are",self.w
            print "Probabilities are",self.p_expert
#        self.beta = self.beta*
        self.counter = self.counter + 1
#        self.beta=

class UserSoftMax(UserBase):
    counter = []
    w = []
    p_expert = []
    n_choice = 0
    n_experts = 0
    beta = 1.0
    eta = 1.0
    gamma = 0.1
    #advice = []
    
    def __init__(self,config):
        UserBase.__init__(self)
        if(config.has_key('n_choice')):
            self.n_choice = config['n_choice']
        else:
            raise KeyError
        if(config.has_key('n_experts')):
            self.n_experts = config['n_experts']
        else:
            raise KeyError
        if(config.has_key('beta')):
            self.beta = config['beta']
        else:
            self.beta = 1.0
        if(config.has_key('eta')):
            self.eta = config['eta']
        else:
            self.eta = 1.0
        if(config.has_key('gamma')):
            self.gamma = config['gamma']
        else:
            self.gamma = 0.1
        #self.w = [1.0]*self.n_experts
        
    def reset(self):
        print "User reset called"
        self.w = array([1.0]*self.n_experts)
        self.counter = array([1.0]*self.n_experts)

    def next_d(self,state):
        if(len(state)==0):
            state = array(self.w)
        a_tmp = exp(-self.eta*state/self.counter)
        #pow(array(self.w),1/(self.counter+1))
        # The following is a mixture SoftMax + Exp3
        #p_expert = (1.0-self.gamma)*a_tmp/a_tmp.sum()+\
        #                self.gamma/self.n_experts
        # The following is pure SoftMax
        p_expert = a_tmp/a_tmp.sum()
        return p_expert

    # This function returns the next choice of the expert
    def next(self):
        # Compute the weighted distribution of experts
        self.p_expert = self.next_d(self.w)
        # Draw an expert due to above distro
        choice = argmax(random.multinomial(1,self.p_expert))
        # Return the expert -- this is the choice that the user makes
        return choice

    # This function updates internal data, based on the user's choice
    def update(self,info):
        # Update all weights based on the info (value of all mentioned experts)
        if (max(self.counter) == 0):
            print "Weights are",self.w/self.counter
            print "Probabilities are",self.p_expert
        val,experts = info
        for exp_idx in experts:
            self.w[exp_idx]=self.w[exp_idx]+val
            self.counter[exp_idx]=self.counter[exp_idx]+1


class UserEGreedy(UserBase):
    counter = []
    w = []
    p_expert = []
    n_choice = 0
    n_experts = 0
    epsilon = 0.1
    #advice = []
    
    def __init__(self,config):
        UserBase.__init__(self)
        if(config.has_key('n_choice')):
            self.n_choice = config['n_choice']
        else:
            raise KeyError
        if(config.has_key('n_experts')):
            self.n_experts = config['n_experts']
        else:
            raise KeyError
        if(config.has_key('epsilon')):
            self.epsilon = config['epsilon']
        else:
            self.epsilon = 0.1
        #self.w = [1.0]*self.n_experts
        
    def reset(self):
        print "User reset called"
        self.w = array([0.0]*self.n_experts)
        self.counter = array([1.0]*self.n_experts)

    def next_d(self,state):
        if(len(state)==0):
            state = array(self.w)
        p_expert = array([self.epsilon/self.n_experts]*self.n_experts)
        a_tmp = state/self.counter
        idx_best = argmin(a_tmp)
        p_expert[idx_best]=p_expert[idx_best]+(1.0-self.epsilon)
        return p_expert

    # This function returns the next choice of the expert
    def next(self):
        # Compute the weighted distribution of experts
        # self.p_expert = array([self.epsilon/self.n_experts]*self.n_experts)
        # a_tmp = self.w/self.counter
        # idx_best = argmin(a_tmp)
        # self.p_expert[idx_best]=self.p_expert[idx_best]+(1.0-self.epsilon)
        self.p_expert = self.next_d(self.w);

        # Draw an expert due to above distro
        choice = argmax(random.multinomial(1,self.p_expert))
        # Return the expert -- this is the choice that the user makes
        return choice

    # This function updates internal data, based on the user's choice
    def update(self,info):
        # Update all weights based on the info (value of all mentioned experts)
        if (max(self.counter) == 0):
            print "Weights are",self.w/self.counter
            print "Probabilities are",self.p_expert
        val,experts = info
        for exp_idx in experts:
            self.w[exp_idx]=self.w[exp_idx]+val
            self.counter[exp_idx]=self.counter[exp_idx]+1
