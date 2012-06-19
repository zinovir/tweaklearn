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

for arg in sys.argv:
    print arg
print len(sys.argv), sys.argv[0]

# %%%%%%
# %import pickle
# %pickle.dump(gen_param,f)
# %pickle.dump(user_param,f)
# %f.close()
# %f = open('config_params_v1_py.pickle','r')
# %gen_param = pickle.load(f)
# %user_param = pickle.load(f)
# %f.close()
