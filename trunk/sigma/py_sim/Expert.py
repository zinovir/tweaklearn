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
        self.advice=argmin(observation[0:self.span])

    def next(self):
        return self.advice
   
