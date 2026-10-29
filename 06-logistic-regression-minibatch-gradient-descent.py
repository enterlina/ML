from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def scale_df(df):
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df


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


def split(df, df_label):
    df_train, df_test, label_train, label_test = train_test_split(df, df_label, test_size=0.6, shuffle=True)
    return df_train, df_test, label_train, label_test


class LogRegression():
    def __init__(self):
        self.weight = None

    def intialize(self, df):
        self.weight =0.001*np.random.rand(df.shape[1])
        # return self.weight

    def loss(self, df, label):
        return np.mean(np.log(1 + np.exp(np.multiply(-label, np.matmul(df, self.weight)))))

    def forward(self, df):
        return (1 / (1 + np.exp(np.matmul(-df, self.weight))))

    def gradient(self, df, label):
        gradient_func = np.divide(label, (1 + np.exp(np.multiply(label, np.matmul(df, self.weight)))))
        gradient_vector = -(np.matmul(df.T, gradient_func)) / df.shape[0]
        self.weight = self.weight - self.learning_rate * gradient_vector
        print('weight', self.weight)
        # return self.weight

    def accuracy(self, df_label, df_pred):
        return accuracy_score(df_label, df_pred)

    def predict(self, df_pred):
        return df_pred > 0.5

    def split_iter(self, df, df_label, steps, k):
        df_split = np.array_split(df, steps)
        label_split = np.array_split(df_label, steps)
        return df_split[k], label_split[k]

    def add_column(self, df):
        scaler = StandardScaler()
        df = np.concatenate([np.ones((df.shape[0], 1)),
                             scaler.fit_transform(df)], axis=1)
        return df

    def regression_iterations(self, df_train, df_test, label_train, label_test, epoch, batches,
                              learning_rate):

        df_train = self.add_column(df_train)

        df_test = self.add_column(df_test)
        self.learning_rate = learning_rate
        self.intialize(df_train)
        self.train_iterations = []
        self.train_accuracy = []
        self.train_loss = []

        self.test_iterations = []
        self.test_accuracy = []
        self.test_loss = []

        for i in range(epoch):
            for k in range(batches):
                self.df_iter, self.label_iter = self.split_iter(df_train, label_train, batches, k)
                # self.df_iter = self.df_iter.astype(float)
                # self.label_iter = self.label_iter.astype(float)
                self.gradient(self.df_iter, self.label_iter)

            self.train_loss.append(self.loss(df_train, label_train))
            self.test_loss.append(self.loss(df_test, label_test))

            self.df_pred_train = self.forward(df_train)
            self.df_pred_test = self.forward(df_test)

            self.train_accuracy.append(self.accuracy(label_train, self.predict(self.df_pred_train)))
            self.test_accuracy.append(self.accuracy(label_test, self.predict(self.df_pred_test)))

            self.train_iterations.append(i)
            self.test_iterations.append(i)

        plt.plot(self.train_iterations, self.test_accuracy)
        plt.ylabel('Error')
        plt.xlabel('Iterations')
        plt.title('Accuracy')
        pyplot.show()

        # plt.plot(self.train_iterations, self.test_loss)
        # plt.ylabel('Error')
        # plt.xlabel('Iterations')
        # plt.title('Loss')
        # pyplot.show()


df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/spam.csv')
# df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/cancer.csv')

df_origin = df_origin.values
df_label = convert(df_origin)
df_label = np.array(df_label)

df_origin = np.array(df_origin)

df_label = np.array(df_label)

df = np.delete(df_origin, 0, 1)
df = np.array(df)
# df = scale_df(df)
batches = 16
epoch = 10000
learning_rate = 1

df_train, df_test, label_train, label_test = split(df, df_label)

regres = LogRegression()

regres.regression_iterations(df_train, df_test, label_train, label_test, epoch, batches, learning_rate)
