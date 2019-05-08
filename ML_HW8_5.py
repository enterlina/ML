#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
import warnings 

import operator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')

from sklearn.linear_model import Lasso





df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW8/noisysine.csv')

x = df_origin['x']
y = df_origin['y']
x = np.array(x)
x = x.reshape(-1, 1)
# y = np.array(y)
# y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, shuffle=True)

for i in range(1, 6):
    alpha=0.01
    x_train_poly = PolynomialFeatures(degree=i).fit_transform(x_train)
    x_test_poly = PolynomialFeatures(degree=i).fit_transform(x_test)
    model = Lasso(alpha=alpha)
    model.fit(x_train_poly , y_train)
    y_poly_pred = model.predict(x_test_poly)
    r_score = r2_score(y_test, y_poly_pred)
    
    for alphanew in np.linspace(0.01, 100, 1000):
        model = Lasso(alpha=alphanew)
        model.fit(x_train_poly, y_train)
        y_poly_pred = model.predict(x_test_poly)
        
        r_newscore = r2_score(y_test, y_poly_pred)
        if r_newscore>r_score:
            alpha=alphanew
            r_score=r_newscore

    
    model = Lasso(alpha=alpha)
    model.fit(x_train_poly, y_train)
    
    y_poly_pred = model.predict(x_test_poly)
    r_score = r2_score(y_test, y_poly_pred)
    print('R-score', r_score, 'Aplha=',alpha, 'Nfeatures=',np.count_nonzero(model.coef_))
    
    plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    
    sorted_zip = sorted(zip(x_test, y_poly_pred ), key=sort_axis)
    x_test, y_poly_pred = zip(*sorted_zip)
    plt.plot(x_test, y_poly_pred, color='m')
    plt.title(f'Linear regression, degree={i}')
    plt.show()
    
  


# In[11]:


df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW8/hydrodynamics.csv')


y=df_origin['y']
x=df_origin.drop(['y'],axis=1)
x=np.array(x)

# x=x.reshape(-1, 1)
# y=np.array(y)
# y=y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, shuffle=True)

for i in range(1, 6):
    alpha=0.01
    x_train_poly = PolynomialFeatures(degree=i).fit_transform(x_train)
    x_test_poly = PolynomialFeatures(degree=i).fit_transform(x_test)
    model = Lasso(alpha=alpha)
    model.fit(x_train_poly , y_train)
    y_poly_pred = model.predict(x_test_poly)
    r_score = r2_score(y_test, y_poly_pred)
    
    for alphanew in np.linspace(0.01, 100, 1000):
        model = Lasso(alpha=alphanew)
        model.fit(x_train_poly, y_train)
        y_poly_pred = model.predict(x_test_poly)
        
        r_newscore = r2_score(y_test, y_poly_pred)
        if r_newscore>r_score:
            alpha=alphanew
            r_score=r_newscore

    
    model = Lasso(alpha=alpha)
    model.fit(x_train_poly, y_train)
    
    y_poly_pred = model.predict(x_test_poly)
    r_score = r2_score(y_test, y_poly_pred)
    print('R-score', r_score, 'Aplha=',alpha, 'Nfeatures=',np.count_nonzero(model.coef_), 'Ntotal=',len(x_test_poly[1,:]) )
    


# In[ ]:





# In[ ]:




