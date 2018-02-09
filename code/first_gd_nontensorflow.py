# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:20:38 2018

@author: Andy
"""
import numpy as np

x = 1.0
learning_rate = 0.1
epochs = 5

def derv(x):
    return 2.0*x*np.exp(x*x)
    return 2.0 * x

print(x)
for _ in range(epochs):
    x = x - learning_rate * derv(x)
print(x)