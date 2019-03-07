import time
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import distance


start = time.time()


def distance(a1, a2):
    eucl_sum = 0
    for i in range(1, len(a2)):
        eucl_sum += (a1[i] - a2[i]) ** 2
    res = np.sqrt(eucl_sum)
    return res


def closer_neighbors(tree_array, array, k_number):
    closer_n = np.zeros((array.shape[0], k_number), dtype=(int, int))
    for i in range(array.shape[0]):
        closer_n[i] = tree_array.query(array.iloc[i], k=k_number)[1]
    return (closer_n)


def kNN(df, cell):
    distance_arr = []
    for i in range(0, df.shape[0]):
        distance_arr.append(distance(cell, df.iloc[i]))
    return (distance_arr)


def detect_class(df, neighbors, var1, var2, k):
    det_class = []
    for i in range(df.shape[0]):
        counter = 0
        for j in range(k):
            if (df['label'].iloc[neighbors[i][j + 1]] == var1):
                counter += 1
        if counter >= k // 2:
            det_class.append(var1)
        else:
            det_class.append(var2)
    return (det_class)


def LOO_Cancer(df, neighbors, var1, var2, k):
    loo_array = []
    for i in range(1, 11):
        res_class = detect_class(df, neighbors, var1, var2, i)
        loo_array.append(1 - len(np.nonzero(np.array(df['label']) == np.array(res_class))[0]) / df.shape[0])
        print('k=', i, '   LOO=', loo_array[i - 1])


def LOO_Spam(df, neighbors, var1, var2, k):
    loo_array = []
    for i in range(1, 11):
        res_class = detect_class(df, neighbors, var1, var2, i)
        loo_array.append(sum(abs(df['label'] - res_class)) / df.shape[0])
        print('k=', i, '   LOO=', loo_array[i - 1])


def RN(df, df_drop, df_tree, var1, var2, R):
    det_class = []
    for i in range(df.shape[0]):
        counter = 0
        neigh = df_tree.query_ball_point(df_drop.iloc[i], r=R)
        for j in neigh:
            if (df['label'].iloc[j] == var1):
                counter += 1
        if counter >= len(neigh) // 2:
            det_class.append(var1)
        else:
            det_class.append(var2)
    return (det_class)


df = pd.read_csv("/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW1/spam.csv")
# df = pd.read_csv("/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW1/cancer.csv")
k = 11

df_drop = df.drop(columns=['label'])
labels = df.label
cell = df.iloc[0]
df_tree = KDTree(df_drop)

euc_distance = kNN(df, cell)
df['distance'] = euc_distance

neighbors = closer_neighbors(df_tree, df_drop, k)
# print(' File: Cancer - Closest 11 \n', df.iloc[neighbors[0]])
# print(' File: Spam - Closest 11 \n', df.iloc[neighbors[0]])

# print(' File: Cancer')
# LOO_Cancer(df, neighbors, 'M', 'B', k)
print(' File: Spam')
# LOO_Spam(df, neighbors, 0, 1, k)

# print(df)
max_dist = df['distance'].max()
print('Max distance = ', max_dist)
# print('Max distance Spam', max_dist)




def LOO_RN(df, df_drop, df_tree, var1, var2, R):
    loo_array = []
    for i in range(1, 30, 4):
        res_class = RN(df, df_drop, df_tree, var1, var2, i)
        loo_array.append(sum(abs(df['label'] - res_class)) / df.shape[0])
        print('R=', i, '   LOO=', loo_array[i//4])


print(LOO_RN(df, df_drop, df_tree, 0, 1, 4))

end = time.time()
print('time =', round(end - start), 's\n')
