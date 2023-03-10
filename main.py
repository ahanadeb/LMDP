import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.gen_trajectories import *
from utils.cluster_assignment import *
from utils.params import *
from utils.acc_ratio import *
from utils.DPMHL import *
import random
import sys
from utils.lmdp import *
from utils.reward import *
from utils.exp_lmdp import *
from utils.util_functions import *


if __name__ == '__main__':
    exp_lmdp()
    #print(get_neighbours(7))
    P_un = uncontrolled_lmdp()
    r= get_reward(F, RF)
    r = reward_feature(M, N, r[0,:])
    z,r1=get_z(r, P_un,30)
    #P=optimal_policy(P_un, z)
    #print(r)
    print(z)
    z, r2 = get_z(r, P_un, 100)
    print(z)
    z, r = get_z(r, P_un, 1000)
    print(z)
    #print(v.shape)
    #print(z.shape)

