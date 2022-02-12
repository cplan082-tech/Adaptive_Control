# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 21:05:47 2022

@author: clive
"""
import numpy as np
import pandas as pd
from math import cos, sin, pi
import scipy as spy
from numpy.linalg import inv
from scipy.signal import unit_impulse 
import matplotlib.pyplot as mpl

# testing vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# df = pd.read_csv('noise_set.csv')
# noise = df['0']
# testing ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sample_depth = 400

a1 = 1.3
a2 = 0.75
b0 = 1.1
b1 = -0.35

theta0 = np.array([a1, a2, b0, b1])
p =100*np.identity(4) # starting P matrix

t = [i for i in range(sample_depth)]
u_t1 = unit_impulse(sample_depth, 100) # Creating impulse delta(t - 100)
u_t2 = np.zeros(sample_depth) # Creating unit step unit(t - 100)
u_t2[np.where(np.arange(0,sample_depth) >= 100)] = 1
u_t3 = np.array([sin(2*pi*t[i]/5) + cos(4*pi*t[i]/5) for i in t])

# u_t = np.stack([u_t1, u_t2]) # impulse = u_t[0], step = u_t[1]
u_t = np.stack([u_t1, u_t2, u_t3]) # impulse = u_t[0], step = u_t[1]

sigma = 0.65
y0 = np.random.normal(0, sigma)
# y0 = np.array(noise[0]) # for testing **************************************
# y = [[y0],[y0]] # first elements in y vector
y = [[y0] for i in range(len(u_t))]

theta_hat0 = np.reshape(np.array([0]*4), (-1,1))
# theta_hat1 = [[theta_hat0], [theta_hat0]]
theta_hat = [[theta_hat0] for i in range(len(u_t))]

#%%

for j in range(len(u_t)):
    p =100*np.identity(4) # starting P matrix
    for i in range(1,sample_depth):
        if (i == 1): # accounts for the lack of t-2 data on first iteration
            phi = np.array([-y[j][i-1], 0, u_t[j][i-1], 0])
        else:
            phi = np.array([-y[j][i-1], -y[j][i-2], u_t[j][i-1], u_t[j][i-2]])
            
        phi = np.asarray(phi).reshape(-1,1) # changes phi's dimensions from (4,) to [4,1] enabling transpose operations
        y[j].append(np.reshape(phi.T@theta0 + np.random.normal(0, sigma), ()))
        # y[j].append(np.reshape(phi.T@theta0 + np.array(noise[i]), ()))
        p = inv(inv(p) + phi*phi.T)
        k = p@phi
        theta_hat[j].append(theta_hat[j][i-1] + k*(y[j][i] - phi.T@theta_hat[j][i-1]))

df_lst = [pd.DataFrame(np.asarray(theta_hat[i]).reshape(-1,4,), 
                   columns=['a1', 'a2', 'b0', 'b1']) for i in range(len(u_t))]

#%%
# mpl.plot(t, df_lst[2]['a1'])


# f, (ax1, ax2, ax3, ax4) = mpl.subplots(4, 1, sharey=True)
# ax1.plot(t, df_lst[2]['a1'])
# ax1.axhline(y=a1, color='black', linestyle='--')

# ax2.plot(t, df_lst[2]['a2'])
# ax2.axhline(y=a2, color='black', linestyle='--')

# ax3.plot(t, df_lst[2]['b0'])
# ax3.axhline(y=b0, color='black', linestyle='--')

# ax4.plot(t, df_lst[2]['b1'])
# ax4.axhline(y=b1, color='black', linestyle='--')



# mpl.plot(t, df_lst[2]['a2'])

# mpl.plot(t, df_lst[2]['b0'])

# mpl.plot(t, df_lst[2]['b1'])



# import seaborn as sns
# sns.set()

# fig, ((ax1, ax2), (ax3, ax4)) = mpl.subplots(2, 2, sharey=True, squeeze=False)
# fig.suptitle('Parameter Extimates for multi-frequency sinusoidial exitation ')
# fig.tight_layout(pad=2)

# ax1.plot(t, df_lst[2]['a1'], label='a1_hat')
# ax1.axhline(y=a1, color='black', linestyle='--', label='a1')
# ax1.legend()
# ax1.set_title('a1')

# ax2.plot(t, df_lst[2]['a2'], label='a2_hat')
# ax2.axhline(y=a2, color='black', linestyle='--', label='a2')
# ax2.legend()
# ax2.set_title('a2')

# ax3.plot(t, df_lst[2]['b0'], label='b0_hat')
# ax3.axhline(y=b0, color='black', linestyle='--', label='b0')
# ax3.legend()
# ax3.set_title('b0')

# ax4.plot(t, df_lst[2]['b1'], label='b1_hat')
# ax4.axhline(y=b1, color='black', linestyle='--', label='b1')
# ax4.legend()
# ax4.set_title('b1')

import matplotlib.pyplot as mpl
import seaborn as sns
line_width = 0.8
graph = sns.lineplot(data=df_lst[2], dashes=False)
# graph.despine(left=True)
graph.axhline(y=a1, color='black', linestyle='--', linewidth=line_width, label='a1')
graph.axhline(y=a2, color='black', linestyle='--', linewidth=line_width, label='a2')
graph.axhline(y=b0, color='black', linestyle='--', linewidth=line_width, label='b0')
graph.axhline(y=b1, color='black', linestyle='--', linewidth=line_width, label='b1')

mpl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labels=df_lst[2].columns)

