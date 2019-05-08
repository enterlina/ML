#!/usr/bin/env python
# coding: utf-8

# In[18]:



import random, numpy, math, copy, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from math import sqrt
from math import exp
from matplotlib import pyplot
from sklearn import metrics
import pandas as pd
import numpy as np
import itertools


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW9/tsp.csv')
y = df['label']
y = np.array(y)
X = df.drop(['label'], axis=1)
X = np.array(X)


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)



iterations=1000

combination_list = []
distance_list = []

while iterations>1:
    distance_comb =0
    change_place = np.random.choice(len(X), size=2, replace=False)
    y[change_place]=y[change_place[::-1]]
    for i in range(1, len(y)):
        distance_comb += distance(X[i- 1], X[i])
        
    distance_list.append(distance_comb)
    combination_list.append(y)
    iterations -=1

path_idx = np.argmin(distance_list)

min_path = distance_list[path_idx]
min_path_combination = combination_list[path_idx]

x_plot = []
y_plot = []
for i in range(1,len(min_path_combination)):
    x_cur = X[min_path_combination[i-1]-1]
    x_next = X[min_path_combination[i]-1]
    x_plot += [x_cur[0], x_next[0]]
    y_plot += [x_cur[1], x_cur[1]]
x_plot += [x_next[0]]
y_plot += [x_next[1]]
print('Min path=', min_path)
# print('Min path combination',min_path_combination)
plt.plot(x_plot, y_plot, 'xb-')
plt.title('Random Walk TSP')
plt.show()


# In[ ]:




