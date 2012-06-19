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
import pickle

if (len(sys.argv)!=3):
    print "Wrong number of arguments. First MBC conf, then policy conf"
    quit()

from User import *
from Expert import *
from MABApp import *

# Loading simulatin parameters
f = open(sys.argv[1]+".pickle",'r');
gen_param = pickle.load(f)
user_param = pickle.load(f)
f.close()

# Loading policy parameters
policy = loadmat(sys.argv[2],mat_dtype=False,squeeze_me=True)
#print policy
#f = open(sys.argv[2]+".pickle",'r');
#f.close()

list_experts = [ExpertConst(0), \
                ExpertConst(1), \
                ExpertConst(2), \
                ExpertConst(3), 
                ExpertWorst(4)]
#                ExpertSelfish(4)]
#                ExpertBest(4)]
#                ExpertCycle(4)]

user = UserSoftMax(user_param)
#UserEGreedy(user_param)
#UserExp3(user_param)
#UserSoftMax(user_param)

list_spa_experts = [ExpertSigma(policy,user),\
                    ExpertWorst(4),\
                    ExpertSelfish(4),\
                    ExpertBest(4)];
list_spa_names = ['sigma','worst','selfish','best'];

#config = dict({'experts':list_experts,\
#               'user':UserCycle(5),\
#               'gen_param':gen_param})

#config = dict({'experts':list_experts,\
#               'user':UserExp3(user_param),\
#               'gen_param':gen_param})

config = dict({'experts':list_experts,\
               'user':user,\
               'gen_param':gen_param})

#config = dict({'experts':list_experts,\
#               'user':UserEGreedy(user_param),\
#               'gen_param':gen_param})

mab_gt = MABGaussTransparent(config)

max_run_time = 600;
exp_no = 75;

for expert in range(len(list_spa_experts)):
    mab_gt['experts'][-1]=list_spa_experts[expert]
    print "Running experiment set for ",list_spa_names[expert]," expert"

    for exp_run_idx in range(exp_no):
        print "\nStarting experiment No:",exp_run_idx,\
              " --- ",list_spa_names[expert]
        history_out = []
        history_exp_poll = []
        history_choice = []
        history_user_info = []
        mab_gt.reset()
        for t_idx in range(max_run_time):
            count,outcome,expert_poll,choice, exp_info, user_info = \
                                              mab_gt.next()
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

        savemat("exp_"+list_spa_names[expert]+"_"+str(exp_run_idx)+".mat",\
                dict({'outcome':history_out,\
                      'polls':history_exp_poll,\
                      'choices':history_choice}))
        
