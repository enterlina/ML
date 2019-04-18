
# coding: utf-8

# In[23]:


# import sklearn.metrics as skmetrics
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import sklearn.model_selection as skms
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def intialize(df):
    weight = 0.01*np.random.sample(len(df[1, :]))
    return weight


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

def loss(df, label, weight):
    return np.mean(np.log(1 + np.exp(np.multiply(-label, (df @ weight)))))


def forward(row, weight):
    logits = row @ weight

    return 1 / (1 + np.exp(-logits))


def gradient(df, label, weight, learning_rate):
    gradient_func = np.divide(label, (1 + np.exp(np.multiply(label, (df @ weight)))))
    gradient_vector = -(df.T @ gradient_func) / len(df)
    weight -= learning_rate * (gradient_vector)
    return weight


def accuracy(df_label, df_pred):
    return accuracy_score(df_label, df_pred)


def predict(row, weights):
    if forward(row, weights) > 0.5:
        return 1
    else:
        return 0


def predict_column(df, weight):
    df_pred = np.zeros((len(df),))
    for i in range(len(df)):
        df_pred[i] = predict(df[i, :], weight)
    return df_pred


def split(df, df_label):
    df_train, df_test, label_train, label_test = train_test_split(df, df_label, test_size=0.4, shuffle=True)
    return df_train, df_test, label_train, label_test


def split_iter(df, df_label, steps, k):
    df_split = np.array_split(df, steps)
    label_split = np.array_split(df_label, steps)
    return df_split[k], label_split[k]


def scale_df(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df




batches = 3
epoch = 1000
learning_rate = 0.5

# df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/spam.csv')
df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/cancer.csv')


df_origin = df_origin.values
df_label = convert(df_origin)
df_label = np.array(df_label)


df_origin = np.array(df_origin)

df_label = np.array(df_label)

df = np.delete(df_origin, 0, 1)
df = np.array(df)


column_one = np.ones((len(df),1))
df=np.append(df,column_one,axis=1)
train_iterations = []
train_accuracy = []
train_loss = []

test_iterations = []
test_accuracy = []
test_loss = []

df = scale_df(df)
df_train, df_test, label_train, label_test = split(df, df_label)

weight = intialize(df)

for i in range(epoch):
    for k in range(batches):
        df_iter, label_iter = split_iter(df_train, label_train, batches, k)
        weight = gradient(df_iter, label_iter, weight, learning_rate)
        
    weight_new=weight
    
    train_loss.append(loss(df_train, label_train, weight_new))
    test_loss.append(loss(df_test, label_test, weight_new))
    
    df_pred_train = predict_column(df_train, weight_new)
    df_pred_test = predict_column(df_test, weight_new)
    
    train_accuracy.append(accuracy(label_train, df_pred_train))
    test_accuracy.append(accuracy(label_test, df_pred_test))
    
    train_iterations.append(i)
    test_iterations.append(i)




# In[22]:


plt.plot(train_iterations, train_loss)
plt.ylabel('Error')
plt.xlabel('Iterations')
pyplot.show()

