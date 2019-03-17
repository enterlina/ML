import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sc
from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances


def calculate_distance(df, df_tree):
    nearest_dist, nearest_ind = df_tree.query(df, k=2)
    return nearest_dist, nearest_ind


def find_min_pairs(nearest_dist, nearest_ind):
    min_dist = []
    nearest = []
    min_dist.append(np.amin(nearest_dist, axis=0)[1])
    for i in range(0, len(nearest_dist)):
        if min_dist in nearest_dist[i]:
            nearest.append(nearest_ind[i])
            i += 1

    return nearest[0]


def dunn_ind(df):
    distance_centr = euclidean_distances(df.groupby(['Cluster']).mean(), df.groupby(['Cluster']).mean())
    distance_array = df.groupby(['Cluster']).apply(lambda x: euclidean_distances(x, x))
    dist_max = []
    for i in range(len(distance_array)):
        dist_max.append(np.max(distance_array.iloc[i]))
    dunn_index = np.min(distance_centr[np.nonzero(distance_centr)]) / np.max(dist_max)
    return dunn_index


def merge_cluster(df, cluster_1, cluster_2):
    df.Cluster[df['Cluster'] == cluster_1] = max(cluster_1, cluster_2)
    df.Cluster[df['Cluster'] == cluster_2] = max(cluster_1, cluster_2)

k = 5
df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW2/blobs2.csv')

Cluster = pd.DataFrame([(lambda x: x)(x) for x in range(0, df.shape[0])])

df['Cluster'] = Cluster
df_original = df.drop(columns='Cluster')
df_tree = KDTree(df_original)

nearest_dist = calculate_distance(df_original, df_tree)[0]
nearest_ind = calculate_distance(df_original, df_tree)[1]
pairs = find_min_pairs(nearest_dist, nearest_ind)


def agglomerative_clustering(df, nearest_dist, nearest_ind, pairs, k):
    distance_matrix = nearest_dist
    min_initialization = nearest_dist[1][1]

    while len(df['Cluster'].unique()) > k:
        cluster_1, cluster_2 = pairs[0], pairs[1]
        merge_cluster(df, cluster_1, cluster_2)
        if len(df['Cluster'].unique()) == k:
            print(df)
            return df


dunn_ind = dunn_ind(df)
# print(df)
agglomerative_clustering(df, nearest_dist, nearest_ind, pairs, k)
print(df)
print('k =', k, 'Dunn_ind =', dunn_ind)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(df[['X']], df[['Y']], c=np.array(df[['Cluster']]), cmap='jet')
plt.show()
