# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 21:05:47 2022

@author: clive
"""

import numpy as np
import scipy as spy
from numpy.linalg import inv
from scipy.signal import unit_impulse 
import matplotlib.pyplot as mpl

sample_depth = 150

a1 = 1.3
a2 = 0.75
b0 = 1.1
b1 = -0.35

theta0 = np.array([a1, a2, b0, b1])
p =np.identity(4) # starting P matrix

sigma = 0.65
y0 = np.random.normal(0, sigma)
y = [[y0],[y0]] # first elements in y vector
# y = [[np.random.normal(0, sigma)],[np.random.normal(0, sigma)]]

u_t1 = unit_impulse(sample_depth, 100) # Creating impulse delta(t - 100)
u_t2 = np.zeros(sample_depth) # Creating unit step unit(t - 100)
u_t2[np.where(np.arange(0,sample_depth) >= 100)] = 1
u_t = np.stack([u_t1, u_t2]) # impulse = u_t[0], step = u_t[1]

theta_hat0 = np.array([0]*4)
theta_hat = [[theta_hat0], [theta_hat0]]


for j in [0, 1]:
    for i in range(1,sample_depth+1):
        if (i == 1): # accounts for the lack of t-2 data on first iteration
            phi = np.array([-y[j][i-1], 0, u_t[j][i-1], 0])
        phi = np.array([-y[j][i-1], -y[j][i-2], u_t[j][i-1], u_t[j][i-2]])
        y[j].append(phi.T@theta0 + np.random.normal(0, sigma))
        p = inv(inv(p) + phi*phi)
        k = p@phi
        theta_hat[j].append(theta_hat[j][i-1] + k*(y[j][i] - phi.T@theta_hat[j][i-1]))

# t = [i for i in range(sample_depth+1)]
# mpl.plot()
        

