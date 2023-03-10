import numpy as np
from tqdm import tqdm
import time
from utils.policies import *
from utils.params import *


def trans(P, pi):  # constructs X by X transition matrix for policy pi
    X = np.size(P, 0)
    P_pi = np.zeros((X, X))
    for m in range(0, X):
        for n in range(0, X):
            P_pi[m][n] = np.dot(P[m][n], pi[m])
    return P_pi


def scalarize(m, n, M, N):  # turns a pair of (m,n) coordinates into a flat 1-dimensional representation x
    x = N * m + n
    return x


def vectorize(x, M, N):  # turns a flat 1-dimensional representation x into a pair of (m,n) coordinates
    n = np.mod(x, N)
    m = (x - n) // N
    return (m, n)


def move(m, n, delta_m, delta_n, M, N,
         obstacles):  # from a given grid position (m,n), move in direction (delta_m,delta_n) if possible
    mtr = np.maximum(np.minimum(m + delta_m, M - 1), 0)
    ntr = np.maximum(np.minimum(n + delta_n, N - 1), 0)
    if obstacles[mtr, ntr] == 1:
        return scalarize(m, n, M, N)
    else:
        return scalarize(mtr, ntr, M, N)


def vector_to_matrix(value_vector, M,
                     N):  # turns a function on the state space represented as a flat X-dimensional vector to an M by N matrix
    X = N * M
    value_matrix = np.zeros((M, N))
    for x in range(0, X):
        [m, n] = vectorize(x, M, N)
        value_matrix[m, n] = value_vector[x]
        if obstacles[m, n] == 1:
            value_matrix[m, n] = np.nan
    return value_matrix


def matrix_to_vector(
        value_matrix):  # turns a function on the state space represented as an M by N matrix into a flat X-dimensional vector
    [M, N] = [np.size(value_matrix, 0), np.size(value_matrix, 1)]
    X = N * M
    value_vector = np.zeros((X))
    for x in range(0, X):
        [m, n] = vectorize(x, M, N)
        value_vector[x] = value_matrix[m, n]
    return value_vector

def evaluate_analytical(P_pi, r, gamma):  ### policy evaluation subroutine (analytical solution)
    X = np.size(P_pi, 0)
    I = np.eye(X)
    A = (I - gamma * P_pi)
    A_inverse = np.linalg.inv(A)
    #r = np.sum(r * pi, axis=1)
    # print("R ", r)
    value = A_inverse.dot(r)
    return value


def conv_policy(policy):
    #converts policy from one hot encoding to Xx1 array
    p = np.zeros((policy.shape[0],1))
    brek
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            if policy[i,j]==1:
                p[i,0]=j

    return p


def get_neighbours(x):
    l=[]
    #grid = np.arange(X).reshape((M,N))
    #up
    if x - N >=0 :
        l.append(x-8)
    #down
    if x +N <=X-1:
        l.append(x+8)
    #left
    if x-1>=0:
        l.append(x-1)
    #right
    if x+1<=X-1:
        l.append(x+1)
    l.append(x)

    return l
