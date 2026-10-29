from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from math import sqrt
from math import exp
from matplotlib import pyplot
from sklearn import metrics
import pandas as pd
import numpy as np


def convert(df):
    df_label = []
    for i in range(len(df)):
        if df[i][0] == 'M':
            df_label.append(1)
        elif df[i][0] == 'B':
            df_label.append(0)
    for i in range(len(df)):
        if df[i][0] == 1:
            df_label.append(1)
        elif df[i][0] == 0:
            df_label.append(0)

    return df_label


def class_prob(y):
    y_dict = collections.Counter(y)
    class_prob = np.ones(2)
    for i in range(0, 2):
        class_prob[i] = y_dict[i] / y.shape[0]

    return class_prob


def mean_variance(X, y):
    n_features = X.shape[1]
    m = np.ones((2, n_features))
    v = np.ones((2, n_features))
    n_0 = np.bincount(y)[np.nonzero(np.bincount(y))[0]][0]
    x0 = np.ones((n_0, n_features))
    x1 = np.ones((X.shape[0] - n_0, n_features))

    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 0:
            x0[k] = X[i]
            k = k + 1
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == 1:
            x1[k] = X[i]
            k = k + 1

    for j in range(0, n_features):
        m[0][j] = np.mean(x0.T[j])
        v[0][j] = np.var(x0.T[j]) * (n_0 / (n_0 - 1))
        m[1][j] = np.mean(x1.T[j])
        v[1][j] = np.var(x1.T[j]) * ((X.shape[0] - n_0) / ((X.shape[0]
                                                            - n_0) - 1))
    return m, v


def prob_feature_class(m, v, x):
    n_features = m.shape[1]
    pfc = np.ones(2)
    for i in range(0, 2):
        product = 1
        for j in range(0, n_features):
            if v[i][j] != 0:
                product = product * (1 / sqrt(2 * 3.14 * v[i][j])) * exp(-0.5 * pow((x[j] - m[i][j]), 2) / v[i][j])

        pfc[i] = product

    return pfc


def naive_bayes(X, y, x):
    m, v = mean_variance(X, y)
    predict_list = []
    pcf_list = []
    for k in range(len(x)):
        pfc = prob_feature_class(m, v, x[k])
        pre_probab = class_prob(y)
        pcf = np.ones(2)
        total_prob = 1
        for i in range(0, 2):
            total_prob = total_prob + (pfc[i] * pre_probab[i])

        for i in range(0, 2):
            pcf[i] = (pfc[i] * pre_probab[i]) / total_prob
            # print('pcf[i]', pcf[i])
        # print('pcf', pcf)
        prediction = int(pcf.argmax())
        predict_list.append(prediction)
        pcf_list.append(pcf[1])
        # print('pcf_list', pcf_list)
    return m, v, pre_probab, pfc, pcf, predict_list, pcf_list


df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW8/spam.csv')

df = df.values
df = np.array(df)
df_label = convert(df)

y = np.array(df_label)
df = np.delete(df, 0, 1)
X = df

df_train, df_test, label_train, label_test = train_test_split(X, y, test_size=0.2, shuffle=True)

mean, variance = mean_variance(df_test, label_test)

m, v, pre_probab, pfc, pcf, predict_list, pcf_list = naive_bayes(df_test, label_test, df_train)

accuracy = accuracy_score(label_train, predict_list)
print('Accuracy', accuracy)
print('ROC-AUC', metrics.roc_auc_score(label_train, predict_list))

r_curve = metrics.roc_curve(label_train, pcf_list)

fpr, tpr, treshold = r_curve

pyplot.plot(fpr, tpr, marker='.')
pyplot.title('ROC-curve')
pyplot.show()
