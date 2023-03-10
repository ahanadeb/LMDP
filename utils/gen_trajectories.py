import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.gen_trajectories import *
from utils.params import *
import random



def traj_form_lmdp(traj):
    traj_states = traj[:, 0]
    states = np.unique(traj_states)
    traj_new = np.zeros((len(states), 2))
    traj_new[:, 0] = states
    for i in range(0, len(states)):
        for j in range(0, len(traj[:, 0])):
            if traj[j, 0] == states[i]:
                traj_new[i, 1] = traj_new[i, 1] + 1  # count of visit
    return traj_new

def traj_merj_lmdp(traj_set, t):
    new = np.zeros((1, 2))
    for s in range(0, X):
        new_arr = np.zeros((1, 2))
        found = 0
        for i in range(len(t)):
            a = traj_set[:, :, int(t[i])]
            a = traj_form_lmdp(a)
            if np.any(a[:, 0] == s):
                j = np.where(a[:, 0] == s)[0][0]
                new_arr[0, 0] = s
                new_arr[0, 1] = new_arr[0, 1] + a[j, 1]
                found = 1
        if found == 1:
            new = np.append(new, new_arr, axis=0)

    return new[1:, :]