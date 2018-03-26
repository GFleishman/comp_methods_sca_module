#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# DEFINE CONVENIENCE FUNCTIONS
def l2_norm(x1, y1, x2, y2):
    return (x2 - x1)**2 + (y2 - y1)**2
# END: DEFINE CONVENIENCE FUNCTIONS


# SET UP THE PROBLEM PARAMETERS
A = (0., 0.)        # left endpoint
B = (1., 0.)        # right endpoint
d = B[0] - A[0]     # distance between fixed endpoints (m)
l = 2*d             # length of chain (m)
g = 9.8             # gravitational constant (m/s^2)
N = 20              # number of discrete points along chain

x = np.linspace(A[0], B[0], N)                   # x-coords of nodes
y = np.linspace(A[1], B[1], N)                   # y-coords or nodes
m = np.ones(N)                                   # mass of each node
dl = np.ones(N-1)*(l/(N-1))**2                   # square edge lengths
# minimize requires single array of variables
x = np.concatenate((x, y))                       # all coords now in x
# END: SET UP THE PROBLEM PARAMETERS


# SET UP OPTIMIZATION OPTIONS
optim_options = {'first-order': True,
                 'SLSQP': True}
# END: SET UP OPTIMIZATION OPTIONS


# DEFINE OBJECTIVE AND CONSTRAINTS
def potential(x):
    return np.sum(m*g*x[N:])

# fixed edge lengths
cons = [{'type': 'ineq',
         'fun': lambda x, i=i: -l2_norm(x[i-1], x[N+i-1],
                                        x[i], x[N+i]) + dl[i-1]}
        for i in range(1, N)]
# fixed endpoints
cons.append({'type': 'eq',
             'fun':  lambda x: np.array(x[0] - A[0])})
cons.append({'type': 'eq',
             'fun':  lambda x: np.array(x[N-1] - B[0])})
cons.append({'type': 'eq',
             'fun':  lambda x: np.array(x[N] - A[1])})
cons.append({'type': 'eq',
             'fun':  lambda x: np.array(x[-1] - B[1])})
# END: DEFINE OBJECTIVE AND CONSTRAINTS


# DEFINE FIRST ORDER DERIVATIVES
if optim_options['first-order']:
    # potential gradient
    def potential_grad(x):
        return np.concatenate((np.zeros(N), np.ones(N)*m*g))
    
    # fixed edge lengths constraint gradient
    diff = np.zeros(N)  # convenient for later
    diff[0] = 2; diff[1] = -2
    for i in range(N-1):
        cons[i]['jac'] = lambda x, i=i: np.concatenate(
                                        (np.roll(diff, i)*(x[i+1]-x[i]),
                                         np.roll(diff, i)*(x[N+i+1]-x[N+i])))
    # fixed endpoint constraint gradients
    delta = np.zeros(2*N)
    delta[0] = 1
    cons[-4]['jac'] = lambda x: delta
    cons[-3]['jac'] = lambda x: np.roll(delta, N-1)
    cons[-2]['jac'] = lambda x: np.roll(delta, N)
    cons[-1]['jac'] = lambda x: np.roll(delta, -1)
# END: DEFINE FIRST ORDER DERIVATIVES


# DEFINE FUNCTION TO DISPLAY OPTIMIZATION ITERATIONS
fig = plt.figure('Optimization Progress')
def display_iteration(xi):
    plt.clf()
    plt.plot(xi[:N], xi[N:], '-o')
    plt.axis('equal')
    plt.pause(.001)
    plt.draw()
# END: DEFINE FUNCTION TO DISPLAY OPTIMIZATION ITERATIONS


# RUN OPTIMIZATION
if optim_options['SLSQP'] and not optim_options['first-order']:
    solution = minimize(potential, x, method='SLSQP',
                        constraints=cons,
                        options={'maxiter': 200,
                                 'disp': True},
                        callback=display_iteration)
if optim_options['SLSQP'] and optim_options['first-order']:
    solution = minimize(potential, x, method='SLSQP',
                        jac=potential_grad,
                        constraints=cons,
                        options={'maxiter': 200,
                                 'disp': True},
                        callback=display_iteration)
# END: RUN OPTIMIZATION


# PLOT FINAL SOLUTION
plt.clf()
plt.plot(solution.x[:N], solution.x[N:], '-o')
plt.axis('equal')
plt.show()
# END: PLOT FINAL SOLUTION
