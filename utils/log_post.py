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
from utils.lmdp import *

np.set_printoptions(threshold=np.inf)


def calLogLLH(r, traj, p1, P_un):
    r = reward_feature(M, N, r).reshape(X, 1)
    z, r2 = get_z(r, P_un)
    #print("here",(z < 0))
    P = optimal_policy(P_un, z)
    #print((P >= 0) & (P <= 1))
    llh = 0
    v = np.log(z)
    for i in range(0, len(traj)):
        tr = traj[i]
        states = tr[:, 0]
        next_s = tr[:, 1]
        for j in range(0, tr.shape[0]):
            llh = llh + np.log(P[int(states[j]), int(next_s[j])])


    return llh


def calLogPrior(r):
    x = r - mu
    prior = np.sum(-1 * (r * np.transpose(r)) / (2 * np.power(sigma, 2)))
    grad = -x / np.power(sigma, 2)
    return prior, grad


def stateFeature(X, F, M, N):
    F = np.zeros((X, F))
    for x in range(1, M + 1):
        for y in range(1, N + 1):
            s = loc2s(x, y, M)
            i = np.ceil(x / B)
            j = np.ceil(y / B)
            f = loc2s(i, j, M / B);
            F[int(s) - 1, int(f) - 1] = 1;
    return F


def loc2s(x, y, M):
    x = max(1, min(M, x));
    y = max(1, min(M, y));
    s = (y - 1) * M + x;

    return s


def calDPMLogPost(traj_set, C, P_un):
    # print(C.assignment)
    logDPprior = np.log(assignment_prob(C.assignment, alpha))
    logLLH = 0
    logPrior = 0
    NC = int(np.max(C.assignment))
    for k in range(0, NC + 1):
        r1 = C.reward[:, k]
        if not C.policy_empty:
            # from reward get best policy and optimal value
            r = reward_feature(M, N, r1).reshape(X, 1)
            z, r2 = get_z(r, P_un)
            # P = get_transitions(M, N, A, p, q, obstacles)
            # V, V_hist, policy, time = policy_iteration(X, P, r, A, gamma, max_iter=100)
            policy = optimal_policy(P_un, z)
        else:
            policy = C.policy[:, :, k]

        t = []
        traj = []
        for l in range(0, len(C.assignment)):
            if C.assignment[l] == k:
                t = np.append(t, l)
        # if len(t) > 1:
        for y in t:
            traj.append(traj_set[:, :, int(y)])
        # else:
        # traj = traj_form_lmdp(traj_set[:, :, int(t[0])])
        # traj = (traj_set[:, :, int(t[0])])
        # print(C.assignment)
        # print("traj len",len(traj))
        llh = calLogLLH(r1, traj, policy, P_un)
        prior, gradP = calLogPrior(r1)
        logLLH = logLLH + llh
        logPrior = logPrior + prior
    print("logpost ", logDPprior, " ", logLLH, " ", logPrior)
    logPost = logDPprior + logLLH + logPrior

    return logPost
