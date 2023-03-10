import numpy as np
from utils.reward import *
from utils.params import *
import scipy as sp
import scipy.sparse
from utils.transition import *
from numpy import linalg as LA

#P_un = uncontrolled transition dynamics
def get_z(r, P_un):
    # np.set_printoptions(threshold=np.inf)
    #r = reward_feature(M, N, r)
    # print(r)
    r = matrix_to_vector(r)
    # print(r)
    z = np.ones((X, 1))
    G = np.zeros((X, X))
    for i in range(X):
        G[i, i] = np.exp(r[i])
    # G = sp.sparse.spdiags(r, 0, r.size, r.size)
    for i in range(0,100):
        z = np.matmul(G, np.matmul(P_un, np.power(z,gamma)))
        #print("z",z)
    w, v = LA.eigh(np.matmul(G,P_un))
    w1=np.argmax(w)
    v = v[w1,:]
    v=v.reshape((X,1))
    if z.any() <0:
        print("problem z")
    return z, r


def optimal_policy(P_un, z):
    a = np.zeros((X, X))
    for i in range(0, X):
        s=0
        for j in range(0, X):
            a[i, j] = P_un[i, j] * z[j]
            s= s+a[i, j]
        #print(np.sum(a[i, :].reshape((1,X)), axis=1))
        a[i, :] = a[i, :] / s
    if a.any() < 0 or a.any() >1:
        print("problem policy")
    return a


def lmdp_gen_traj(X, P, tl):
    traj = np.zeros((tl, 2))
    s0= random.randint(0, X - 1)  # initial state

    for i in range(0, tl):
        states = (np.arange(X)).reshape((X,))
        traj[i, 0] = s0
        next_s = random.choices(states, weights=P[s0, :].reshape((X,)), k=1)
        #next_s = [np.argmax(P[s0, :])]
        traj[i, 1] = next_s[0]
        s0=next_s[0]
    #print("traj", traj)
    return traj


def lmdp_trajectories(F, P_un, RF, tn, tl):
    r = get_reward(F, RF)  # shape RF*F 3*16
    traj_data = np.zeros((tl, 2, tn))  # 6 trajectories
    seq = []
    traj_per_agent = int(tn / RF)
    y = np.arange(RF)
    y = np.repeat(y, traj_per_agent)
    np.random.shuffle(y)

    for i in range(0, tn):
        j = int(y[i])
        seq.append(j)
        reward = r[j, :]
        reward = reward_feature(M, N, reward).reshape(X, 1)
        z, r2 = get_z(reward, P_un)
        P = optimal_policy(P_un, z)
        traj = lmdp_gen_traj(X, P, tl)
        traj_data[:, :, i] = traj
    rewards_gt = np.zeros((F, 1))
    for k in range(len(seq)):
        rewards_gt = np.append(rewards_gt, np.transpose(r[j, :]).reshape((F, 1)), axis=1)
    rewards_gt = rewards_gt[:, 1:]
    #print("now", traj_data.shape)
    print("original assignment", y)
    return traj_data, rewards_gt,y
