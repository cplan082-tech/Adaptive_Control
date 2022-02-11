# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 22:17:10 2022

@author: clive
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl

sigma = 0.65

sample_set = [np.random.normal(0, sigma) for i in range(400)]
t = [i for i in range(400)]
mpl.plot(t,sample_set)

df = pd.DataFrame(sample_set)
df.to_csv('noise_set.csv')
print(len(df))