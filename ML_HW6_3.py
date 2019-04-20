
# coding: utf-8

# In[58]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/mnist.csv')
df_label = df['label']
df = df.values

df = np.array(df)

df = np.delete(df, (0), axis=1)
df_norm = df / 255
df_norm.mean(axis=0).shape

df_norm = df_norm - df_norm.mean(axis=0)

data = []

for i in range(0, df_norm.shape[0]):
    data.append(np.array(df_norm[i, :]).reshape(28, 28))
data = np.array(data)

df_train, df_test, label_train, label_test = train_test_split(data, df_label, test_size=0.2, shuffle=True)


def cnn_model(value):
    model = Sequential()

    model.add(Conv2D(8, (3, 3), activation='sigmoid', input_shape=(28, 28, 1)))
    model.add(Conv2D(8, (3, 3), activation='tanh'))
    model.add(Conv2D(8, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(64, activation=value))
    model.add(Dense(64, activation=value))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


df_train = df_train.reshape(df_train.shape[0], 28, 28, 1)
label_train = tf.keras.utils.to_categorical(label_train)
df_test = df_test.reshape(df_test.shape[0], 28, 28, 1)
label_test = tf.keras.utils.to_categorical(label_test)


# In[62]:


model = cnn_model('sigmoid')

model.fit(df_train, label_train, epochs = 1)
score = model.evaluate(df_test, label_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])


# In[63]:


model = cnn_model('tanh')

model.fit(df_train, label_train, epochs = 1)
score = model.evaluate(df_test, label_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])


# In[64]:


model = cnn_model('relu')

model.fit(df_train, label_train, epochs = 1)
score = model.evaluate(df_test, label_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])

