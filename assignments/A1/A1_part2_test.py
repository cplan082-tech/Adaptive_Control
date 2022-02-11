# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 21:34:43 2022

@author: clive
"""
import pandas as pd
import numpy as np
import scipy as spy
from numpy.linalg import inv
from scipy.signal import unit_impulse 
import matplotlib.pyplot as mpl


# df = pd.read_csv('noise_set.csv')
# noise = df['0']

sample_depth = 400

a1 = 1.3
a2 = 0.75
b0 = 1.1
b1 = -0.35

theta0 = np.array([a1, a2, b0, b1])
p =100*np.identity(4) # starting P matrix

sigma = 0.65
# y = [np.array([noise[0]])]# first elements in y vector
y = [np.random.normal(0, sigma)]


u_t = unit_impulse(sample_depth, 100) # Creating impulse delta(t - 100)


theta_hat = [np.reshape(np.array([0]*4), (-1,1))]



for i in range(1,sample_depth):
    # print(i)
    if (i == 1): # accounts for the lack of t-2 data on first iteration
        phi = np.array([-y[i-1], 0, u_t[i-1], 0])
    else:
        phi = np.array([-y[i-1], -y[i-2], u_t[i-1], u_t[i-2]])
        
    phi = np.asarray(phi).reshape(-1,1)
    y_temp = np.reshape(phi.T@theta0 + np.random.normal(0, sigma), ()) # why did I have to reshape here!?!?!??!
    y.append(y_temp)
    p = inv(inv(p) + phi*phi.T)
    k = p@phi
    theta_hat.append(theta_hat[i-1] + k*(y[i] - phi.T@theta_hat[i-1]))

# t = [i for i in range(sample_depth+1)]
# mpl.plot(y)
        

