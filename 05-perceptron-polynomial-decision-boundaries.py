
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt


def predict(row, weights):
    if np.matmul(row, weights) >= 0:
        return 1
    else:
        return 0


def predict_column(df, weights):
    df_pred = np.zeros((len(df),))
    for i in range(len(df)):
        df_pred[i] = predict(df[i, :], weights)
    return df_pred


def accurancy(df_label, df_pred):
    return accuracy_score(df_label, df_pred)


def perceptron(df, df_label, rate, epoch):
    auc_dict = []
    weights = []
    weight = np.random.sample(len(df[1, :]))
    for i in range(epoch):
        for j in range(len(df)):
            df_pred = predict_column(df, weight)
            if (df_label[j] == 1) and (df_pred[j] == 0):
                weight = weight + df[j]
            elif (df_label[j] == 0) and (df_pred[j] == 1):
                weight = weight - df[j]
            else:
                weight = weight
            auc_dict.append(accurancy(df_label, df_pred))
            weights.append(weight)
        i += 1
    return auc_dict, weights


def graph(X, y,n, weight_best):
    h = 0.04
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = predict_column(PolynomialFeatures(degree=n).fit_transform(np.c_[xx.ravel(), yy.ravel()]), weight_best)
    Z = Z.reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.contour(xx, yy, Z, colors=['b', 'b'])


n_folds = 3
rate = 0.5
epoch = 100

df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW5/blobs2.csv')
df_origin = df_origin.values
df_origin = np.array(df_origin)

df_label = df_origin[:, 2]
df_label = np.array(df_label)
df = np.delete(df_origin, 2, 1)
df = np.array(df)
n = 4
df_poly = PolynomialFeatures(degree=n).fit_transform(df)

auc_dict, weights = perceptron(df_poly, df_label, rate, epoch)
auc_dict = np.array(auc_dict)
idx = auc_dict.argmax()
weight_best = weights[idx]

# df_pred = predict_column(df_poly, weight_best)

print('n=', n)
print('acc=', auc_dict[idx])
print('weights=', weights[idx])


graph(df, df_label, n, weight_best)
