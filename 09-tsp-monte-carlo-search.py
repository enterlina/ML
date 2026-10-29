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
combination_list = []
distance_list = []

for k in range(1000000):
    combination = np.random.permutation(len(y))

    distance_comb = 0
    for i in range(1, len(combination)):
        distance_comb += distance(X[combination[i - 1] - 1], X[combination[i] - 1])

    distance_list.append(distance_comb)
    combination_list.append(combination)
path_idx = np.argmin(distance_list)

min_path = distance_list[path_idx]
min_path_combination = combination_list[path_idx]

print('Min path=', min_path)
# print(min_path_combination)
x_plot = []
y_plot = []
for i in range(1, len(min_path_combination)):
    x_cur = X[min_path_combination[i - 1] - 1]
    x_next = X[min_path_combination[i] - 1]
    x_plot += [x_cur[0], x_next[0]]
    y_plot += [x_cur[1], x_cur[1]]
x_plot += [x_next[0]]
y_plot += [x_next[1]]
plt.plot(x_plot, y_plot, 'xb-')
plt.title('Monte Carlo TSP')
plt.show()
