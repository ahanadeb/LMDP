import numpy as np
from utils.util_functions import *
from utils.acc_ratio import *
import random
from scipy.stats import dirichlet
from scipy.special import factorial
from utils.params import *
from utils.reward import *
from utils.transition import *
from utils.gen_trajectories import *
from utils.cluster_assignment import *
from utils.log_post import *
from utils.saveHist import *
from utils.lmdp import *


def evd(hist, reward_gt, maxiter, tn, P_un,y):
    # print("hist length", len(hist.policy))
    e = 0
    for i in range(0, tn):
        k = int(hist.assignment[i])
        # changed k to i here in the next line
        r1 = reward_feature(M, N, reward_gt[:, i]).reshape(X, 1)
        print("association", k, y[i])
        r2 = hist.reward[:, k]
        r2 = reward_feature(M, N, r2).reshape(X, 1)

        z_eval, r_n = get_z(r2, P_un)
        opt_policy = optimal_policy(P_un, z_eval)
        V_eval = evaluate_analytical(opt_policy, r1, gamma)



        z_true , r_m = get_z(r1, P_un)
        opt_policy = optimal_policy(P_un, z_true)
        V_true = evaluate_analytical(opt_policy, r1, gamma)

        e = e + (V_true - V_eval)
    e = e / tn
    #print("v_true", V_true)
    #print("v_eval", V_eval)
    #print(np.mean(e))
    return np.mean(e)
