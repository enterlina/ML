
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

import time
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets



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


# df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW4/spam.csv')
df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW7/cancer.csv')


df = df.values
df = np.array(df)
df_label=convert(df)

y = np.array(df_label)
df = np.delete(df, 0, 1)
X = df

df_train, df_test, label_train, label_test = train_test_split(X, y, test_size=0.2, shuffle=True)

C = 1.0
model = svm.SVC(kernel='linear', C=C)
model.fit(df_train, label_train)
pred = model.predict(df_test)

start = time.clock()

print(f"Time: {time.clock() - start}")
print(f"Accuracy: {accuracy_score(label_test, pred)}")


# In[14]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(df_train, label_train)
pred = model.predict(df_test)

start = time.clock()
print(f"Time: {time.clock() - start}")
print(f"Accuracy: {accuracy_score(label_test, pred)}")

