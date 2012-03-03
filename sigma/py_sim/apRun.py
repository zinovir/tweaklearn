# Load full scientific calculations support
from numpy import *
import scipy as Sci
import scipy.linalg
# Ensure you can save Matlab formated trace datafiles
from scipy.io import *
#import random

# Ensure you can work with files and system services
import sys
# Ensure basic math is available
import math

from User import *
from Expert import *
from MABApp import *

list_experts = [ExpertConst(0), \
                ExpertConst(1), \
                ExpertConst(2), \
                ExpertConst(3), \
                ExpertBest(4)]
#                ExpertCycle(4)]

gen_param = dict({'mu':array([3.0,2.0,1.5,4.0,2.0,1.5,2.0,1.5]),\
                  'sigma':array([[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],\
                                 [0.0,0.7,0.0,0.0,0.0,0.0,0.0,0.0],\
                                 [0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],\
                                 [0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0],\
                                 [0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],\
                                 [0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],\
                                 [0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],\
                                 [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]),\
                  'discr':4.0,\
                  'n_choice':4})

user_param = dict({'n_experts':5,\
                   'n_choice':4,\
                   'beta':0.01,\
                   'gamma':0.1,\
                   'beta_spec':0.05,\
                   'eta':15.0,\
                   'epsilon':0.05})

config_saver = dict(gen_param)
for x in user_param.keys():
    config_saver[x]=user_param[x]
savemat("config_params_v1",config_saver)

#config = dict({'experts':list_experts,\
#               'user':UserCycle(5),\
#               'gen_param':gen_param})

#config = dict({'experts':list_experts,\
#               'user':UserExp3(user_param),\
#               'gen_param':gen_param})

config = dict({'experts':list_experts,\
               'user':UserSoftMax(user_param),\
               'gen_param':gen_param})

#config = dict({'experts':list_experts,\
#               'user':UserEGreedy(user_param),\
#               'gen_param':gen_param})

mab_gt = MABGaussTransparent(config)

max_run_time = 600;
exp_no = 75;

for exp_run_idx in range(exp_no):
    print "\nStarting experiment No:",exp_run_idx
    history_out = []
    history_exp_poll = []
    history_choice = []
    history_user_info = []
    mab_gt.reset()
    for t_idx in range(max_run_time):
        count,outcome,expert_poll,choice, exp_info, user_info = mab_gt.next()
        history_out.append(outcome)
        history_exp_poll.append(expert_poll)
        history_choice.append(choice)
        history_user_info.append(user_info)
        if ((t_idx == 0) or (t_idx == (max_run_time-1))):
            print "Time step ",count
            print "Outcome ", outcome
            print "Expert poll ",expert_poll
            print "Choice made", choice
            print "Information fed back ", user_info
            #print "\n"
    
    savemat("tmp"+str(exp_run_idx)+".mat",dict({'outcome':history_out,\
                            'polls':history_exp_poll,\
                            'choices':history_choice}))
