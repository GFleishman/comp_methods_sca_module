#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# SET UP THE PROBLEM PARAMETERS
A = (0., 0.)        # left endpoint
B = (1., 0.)        # right endpoint
d = B[0] - A[0]     # distance between fixed endpoints (m)
l = 2*d             # length of chain (m)
g = 9.8             # gravitational constant (m/s^2)
N = 20              # number of discrete points along chain
mu = 1e6            # initial penalty weight: 1e6 good for SLSQP and BFGS
iterations = 4      # number of outer loop iterations in optimization

x = np.arange(1, N-1)*(B[0] - A[0])/(N-1)        # x-coords of movable nodes
y = np.arange(1, N-1)*(B[1] - A[1])/(N-1)        # y-coords of movable nodes
lam = np.zeros(N-1)                              # array of Lagrange mult
m = np.ones(N-2)                                 # mass of each node
dl = np.ones(N-1)*l/(N-1)                        # edge lengths
# minimize requires single array of variables
x = np.concatenate((x, y, lam))                 # all coords and lams now in x
# END: SET UP THE PROBLEM PARAMETERS


# SET UP OPTIMIZATION OPTIONS
optim_method = 'BFGS'
# method options: 'SLSQP'
#                 'nelder-mead' 
#                 'BFGS'
# END: SET UP OPTIMIZATION OPTIONS


# DEFINE CONVENIENCE FUNCTIONS
def l2_norm(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# END: DEFINE CONVENIENCE FUNCTIONS


# DEFINE CONVENIENCE FUNCTION FOR COMPUTING LAGRANGIAN
def link_distances(x, A=A, B=B):
    # identify params (just for readability)
    x_ = x[:N-2]
    y_ = x[N-2:2*(N-2)]
    return np.array( [l2_norm(A[0], A[1], x_[0], y_[0])] +
                     list(l2_norm(x_[:-1], y_[:-1], x_[1:], y_[1:])) +
                     [l2_norm(x_[-1], y_[-1], B[0], B[1])] )
# END: DEFINE CONVENIENCE FUNCTION FOR COMPUTING LAGRANGIAN


# DEFINE LAGRANGIAN
def lagrangian(x, mu):
    # identify params (just for readability)
    y_ = x[N-2:2*(N-2)]
    lam_ = x[2*(N-2):]
    # compute potential and link constraint terms
    potential = np.sum(m*g*y_)
    link_dists = link_distances(x)
    link_con = np.sum(lam_*(link_dists - dl))
    augmentation = mu * np.sum((link_dists - dl)**2)
    return potential - link_con + augmentation
# END: DEFINE LAGRANGIAN


# DEFINE FUNCTION TO DISPLAY OPTIMIZATION ITERATIONS
fig = plt.figure('Optimization Progress')
def display_iteration(xi):
    plt.clf()
    plt.plot(xi[:N-2], xi[N-2:2*(N-2)], '-o')
    plt.axis('equal')
    plt.pause(.001)
    plt.draw()
# END: DEFINE FUNCTION TO DISPLAY OPTIMIZATION ITERATIONS


# RUN OPTIMIZATION
for i in range(iterations):
    solution = minimize(lagrangian, x, args=(mu,),
                        method=optim_method,
                        options={'maxiter': 200,
                                 'disp': True},
                        callback=display_iteration)
    x = solution.x
    constraint_eqs = link_distances(x) - dl
    # update lams according to augmented Lagrangian method update rule
    x[2*(N-2):] = x[2*(N-2):] - mu*constraint_eqs
    # increase mu: ensures stable and proper regularization
    mu *= 2
# END: RUN OPTIMIZATION


# REPORT RESULTS
plt.clf()
plt.plot(x[:N-2], x[N-2:2*(N-2)], '-o')
plt.axis('equal')
plt.show()
# END: REPORT RESULTS
# END SCRIPT
