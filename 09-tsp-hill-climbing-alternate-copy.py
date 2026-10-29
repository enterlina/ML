#!/usr/bin/env python
# coding: utf-8

# In[9]:




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

from scipy.spatial.distance import cityblock

def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW9/tsp.csv')
y = df['label']
y = np.array(y)
X = df.drop(['label'], axis=1)
X = np.array(X)
min_path_list=[]
n = 100

min_combination_list=[]



for k in range(n ):
    combination = np.random.permutation(len(y))
    
    current_dist=0
    for i in range(1, len(combination)):
        current_dist += cityblock(X[combination[i - 1]], X[combination[i]])

    min_path = current_dist
    tag=1
    while tag:
        combination_list = []
        distance_list = []
        tag=0

        for i in range(len(combination) - 2):
            for j in range (i+1, len(combination) - 1):

                neighbor = np.copy(combination)
                distance_comb = 0
                combination_list = []
                distance_list = []
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
                for l in range(1, len(neighbor)):
                    distance_comb += cityblock(X[neighbor[l - 1]], X[neighbor[l]])

                distance_list.append(distance_comb)
                combination_list.append(neighbor)

                path_idx = np.argmin(distance_list)
 
                if distance_list [path_idx]< min_path:
                    combination= neighbor
                    min_path = distance_list[path_idx]
                    min_path_list.append(min_path)
                    min_combination_list.append(combination)
                    tag=1

path_idx = np.argmin(min_path_list)
min_path=min_path_list[path_idx]
min_path_combination = min_combination_list[path_idx]

print('Min path=', min_path)
# print('min_path_combination',min_path_combination)


way = X[min_path_combination,:]
x, y = [], []
x.append(way[0, 0])
y.append(way[0, 1])
for i in range(1, way.shape[0]):
    x.append(way[i - 1, 0])
    y.append(way[i, 1])
    x.append(way[i, 0])
    y.append(way[i, 1])

plt.plot(way[:, 0], way[:, 1], 'xb', x, y, 'b')
plt.title('Hill Climb')
plt.show()
    
    


# In[ ]:




