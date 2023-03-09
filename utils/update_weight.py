import numpy as np
from utils.util_functions import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.log_post import *
from utils.acc_ratio import *


def update_weight(k, traj_set, C, P_un):
    for i in range(0,10):
        for i in range(0, weight_iter):
            t = []
            traj=[]
            for l in range(0, len(C.assignment)):
                if C.assignment[l] == k:
                    t = np.append(t, l)
            for y in t:
                traj.append(traj_set[:, :, int(y)])
            #if len(t) > 1:
              #  traj = traj_merj_lmdp(traj_set, t)
            #else:
                #traj = traj_form_lmdp(traj_set[:, :, int(t[0])])
            r1 = C.reward[:, k]
            p1 = C.policy[:, :, k]
            z1 = C.value[:, k]

            r2=sample_reward(F, mu, sigma, lb, ub)
            rx = reward_feature(M, N,r2).reshape(X, 1)
            z2, r3 = get_z(rx, P_un)
            p2 = optimal_policy(P_un, z2)


            ratio = acc_ratio(traj, r2, z2, r1, z1,P_un)
            rand_n = random.uniform(0, 1)
            if rand_n < ratio:
                C.reward[:, k] = np.squeeze(r2)
                C.policy[:,:, k] = p2
                C.value[:, k] = np.squeeze(z2)


    return C
