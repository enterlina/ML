# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances

global q
q = 0.03


def purity(df):
    cluster_list = df.groupby(['Cluster', 'label']).size().reset_index(name='size')
    cluster_list_sum = cluster_list.groupby(['Cluster']).agg({'size': ['max', 'sum']})

    purity_v = sum(cluster_list_sum['size']['max'] / cluster_list_sum['size']['sum']) / len(cluster_list_sum)

    return (float(purity_v))

#
# def purity_normalize(df):
#     cluster_list = df.groupby(['Cluster']).size().reset_index(name='size')
#     cluster_list_sum = cluster_list.groupby(['Cluster']).agg({'size': ['max', 'sum']})
#
#     purity_v = sum(cluster_list_sum['size']['max'] / cluster_list_sum['size']['sum']) / len(cluster_list_sum)
#
#     return (float(purity_v))


def get_clusters(df, df_centroid, df_origin, k):
    for i in range(1, df.shape[0]):
        distance = []
        for j in range(0, k):
            distance.append(euclidean(df_origin.iloc[i], df_centroid.iloc[j]))
        df['Cluster'].iloc[i] = np.argmin(distance)
    return df


def dunn_ind(df):
    distance_centr = euclidean_distances(df.groupby(['Cluster']).mean(), df.groupby(['Cluster']).mean())
    distance_array = df.groupby(['Cluster']).apply(lambda x: euclidean_distances(x, x))
    dist_max = []
    for i in range(len(distance_array)):
        dist_max.append(np.max(distance_array.iloc[i]))
    dunn_index = np.min(distance_centr[np.nonzero(distance_centr)]) / np.max(dist_max)
    return dunn_index


def kMeans(df, centroids, df_origin, dunn_index, alpha, k):
    df_1 = df.copy()
    previous_dunn_index = -100
    new_centroids = df_1.groupby(['Cluster']).mean()

    while (abs(dunn_index - previous_dunn_index) < alpha):
        previous_dunn_index = dunn_index
        centroids = new_centroids.copy()
        df_1 = get_clusters(df_1, centroids, df_origin, k)
        new_centroids = df_1.groupby(['Cluster']).mean()
        df_1['Clusters'] = get_clusters
        dunn_index = dunn_ind(df, centers)

    return (df, new_centroids, dunn_index)


df_1 = pd.read_csv("/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW2/cancer2.csv")

df = df_1.copy()

df = df.drop(columns=['label'])
df = pd.DataFrame(MinMaxScaler().fit_transform(df))
df['Cluster'] = 0
df = df.join(df_1['label'])

# print(df)


df_origin = df.copy
df_origin = df.drop(columns=['Cluster'])

df_origin = df.drop(columns=['Cluster', 'label'])

new_centers = df.groupby(['Cluster']).mean()


q = 0.03

for k in  range(2,11):
    df_centroid = df_origin.iloc[0:k]

    df = get_clusters(df, df_centroid, df_origin, k)

    df_2 = df.drop(columns=['label'])
    dunn = dunn_ind(df_2)

    df = kMeans(df, df_centroid, df_origin, dunn, q, k)[0]
    centers = kMeans(df, df_centroid, df_origin, dunn, q, k)[1]
    print('k =', k, '    purity_normalized =', purity(df))
