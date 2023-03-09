import numpy as np
from utils.lmdp import *
from utils.reward import *
from utils.transition import *
import matplotlib.pyplot as plt
from utils.DPMHL import *
from utils.evd import *


def exp_lmdp():
    print("hey")
    P_un = uncontrolled_lmdp()
    np.set_printoptions(threshold=np.inf)

    EVD = []
    y = []
    i = 2
    while i < 7:
        maxiter = 30
        tn = RF * i
        y.append(int(i))
        traj_set, rewards_gt, y2 = lmdp_trajectories(F, P_un, RF, tn, tl)
        maxC = dpmhl(traj_set, maxiter, tn, P_un)
        e = evd(maxC, rewards_gt, maxiter, tn, P_un, y2)
        EVD.append(e)
        print("EVD = ", EVD)
        i = i + 2
    print("Completed. EVD = ", EVD)
    plt.plot(y, np.asarray(EVD))
    plt.xlabel('no. of trajectories per agent')
    plt.ylabel('EVD for the new trajectory')
    plt.savefig('figure.png')
    plt.show()
    print("np")
