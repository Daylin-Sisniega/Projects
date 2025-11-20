# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 19:44:04 2025

@author: dayli
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-1,1,100)

#print(x)

np.random.seed(45)

y = np.array([],dtype=int)

for i in range(len(x)):
    r = 12*x[i]-4
    y = np.append(y,r)
    

plt.scatter(x, y, alpha=0.5, marker = '*')
plt.title("Sampling and noise")
plt.xlabel("x values")
plt.ylabel("y values")

noise = np.random.normal(3, 2.5, 100)

yn = np.array([],dtype=int)

for j in range(len(y)):
    rn = y[j] + noise[j]
    yn = np.append(yn,rn)

plt.scatter(x, yn, alpha=0.5, marker = '.')
plt.title("Sampling and noise")
plt.xlabel("x values")
plt.ylabel("y values with noise")