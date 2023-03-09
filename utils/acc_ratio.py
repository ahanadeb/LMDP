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
    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]
        for j in range(0, tr.shape[0]):
            a = a + np.log(P1[int(states[j]), int(next_s[j])]) - np.log(P2[int(states[j]), int(next_s[j])])
    ratio = np.exp(a)

    # x = 0
    # for t in range(0, traj.shape[0]):
    #    s = int(traj[t, 0])
    #    #a = int(traj[t, 1])
    #    x = x + traj[t, 1] * (nq1[s] - nq2[s])
    # ratio = np.exp(x)
    # print("ratio",ratio)
    return ratio
