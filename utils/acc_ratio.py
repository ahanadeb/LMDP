import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.lmdp import *


def acc_ratio(traj, r1, z1, r2, z2, P_un):
    P1 = optimal_policy(P_un, z1)
    P2 = optimal_policy(P_un, z2)

    a = 0
    b = 0
    v1= np.log(z1)
    v2 = np.log(z2)
    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]
        states1 = tr[:, 0]
        states2 = tr[:, 1]
        for j in range(0, tr.shape[0]):
            x = 0
            y = 0
            n = get_neighbours(int(states1[j]))
            # print("n", n)
            for k in range(len(n)):
                x = x + np.exp(P1[int(states1[j]),n[k]])*np.exp(v1[n[k]])
                y = y + np.exp(P2[int(states1[j]),n[k]])*np.exp(v2[n[k]])
                # print(x)
            #a = a + P1[int(states1[j]),int(states2[j])]*v1[int(states2[j])] - np.log(x)
           # b = b + P1[int(states1[j]),int(states2[j])]*v2[int(states2[j])] - np.log(y)
            a = a + np.log(P1[int(states[j]), int(next_s[j])])
            b  = b +np.log(P2[int(states[j]), int(next_s[j])])
        #a = a-b
    ratio = np.exp(a-b)
    #print(ratio)

    return ratio
