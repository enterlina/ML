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


def annealing(current_dist, new_dist, t):
    if new_dist < current_dist:
        return 1
    else:
        return np.exp(-(new_dist - current_dist) / t)


df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW9/tsp.csv')
y = df['label']
y = np.array(y)
X = df.drop(['label'], axis=1)
X = np.array(X)
combination_list = []
distance_list = []
current_dist = 0
combination = np.random.permutation(len(y))
n = 10000

for i in range(1, len(combination)):
    current_dist += cityblock(X[i - 1], X[i])

for j in range(n):
    t = n / (j + 1)
    distance_comb = 0
    neighbor = np.copy(combination)
    change_place = np.random.choice(len(X), size=2, replace=False)
    neighbor[change_place] = neighbor[change_place[::-1]]
    for l in range(1, len(neighbor)):
        distance_comb += cityblock(X[neighbor[l - 1]], X[neighbor[l]])

    new_dist = distance_comb

    aneealing_v = annealing(current_dist, new_dist, t)
    if aneealing_v >= np.random.rand():
        combination = neighbor
        current_dist = new_dist

print('Min path=', current_dist)
x_plot = []
y_plot = []
for i in range(1, len(combination)):
    x_cur = X[combination[i - 1] - 1]
    x_next = X[combination[i] - 1]
    x_plot += [x_cur[0], x_next[0]]
    y_plot += [x_cur[1], x_cur[1]]
x_plot += [x_next[0]]
y_plot += [x_next[1]]
plt.plot(x_plot, y_plot, 'xb-')
plt.title('Annealing')
plt.show()
