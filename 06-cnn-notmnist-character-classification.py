
# coding: utf-8

# In[38]:


import os
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np

from sklearn import preprocessing

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def load_image(filename):
    if filename != '/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/notMNIST_small/A/.DS_Store' and filename != '/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/notMNIST_small/F/.DS_Store':
        img = Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data


path = '/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW6/notMNIST_small'
df_label=[]
files = []

for dirname, dirnames, filenames in os.walk(path):

    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        dirname_new = dirname.split(os.path.sep)[-1]
        df_label.append(dirname_new)
df = []
for filename in files:
    df.append(load_image(filename))

df = np.array(df)

df = np.delete(df, (0), axis=0)
df_label= np.delete(df_label, (0), axis=0)
df = np.delete(df, (0), axis=0)
df_label= np.delete(df_label, (0), axis=0)
df_label=list(df_label)

le = preprocessing.LabelEncoder()
le.fit(df_label)
df_label=le.transform(df_label)

df_train, df_test, label_train, label_test = train_test_split(df, df_label, test_size=0.2, shuffle=True)

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

df_train=np.array(df_train)
df_test=np.array(df_test)

df_train = df_train.reshape(df_train.shape[0], 28, 28, 1)
label_train = tf.keras.utils.to_categorical(label_train)
df_test = df_test.reshape(df_test.shape[0], 28, 28, 1)
label_test = tf.keras.utils.to_categorical(label_test)



# In[37]:


model = cnn_model('relu')

model.fit(df_train, label_train, epochs = 1)
score = model.evaluate(df_test, label_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])

