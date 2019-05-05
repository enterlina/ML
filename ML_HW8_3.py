#!/usr/bin/env python
# coding: utf-8

# In[26]:



import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import operator

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures



class LinReg():
    def __init__(self, deg=1, alpha=0):
        self.w = 0
        self.alpha = alpha
        self.polytransform = PolynomialFeatures(degree=deg)
    
    def fit(self, X, y):
        X_poly = self.polytransform.fit_transform(X)
        if self.alpha == 0:
            self.w = np.linalg.pinv(X_poly) @ y
        else:
            self.w = np.linalg.inv(X_poly.T @ X_poly + self.alpha * np.eye(X_poly.shape[1]))                     @ X_poly.T @ y
        
    def predict(self, X):
        X_poly = self.polytransform.fit_transform(X)
        y_pred = X_poly @ self.w
        return y_pred


    
    
df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW8/noisysine.csv')

x=df_origin['x']
y=df_origin['y']
x=np.array(x)
x=x.reshape(-1, 1)
y=np.array(y)
y=y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


for i in range(1,6):
    polynomial_features= PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(x)
    model = LinReg()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r_score = r2_score(y,y_poly_pred)
    print('R-score',r_score)

    plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.title(f'Linear regression, degree={i}')
    plt.show()
    


# In[32]:


df_origin = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW8/hydrodynamics.csv')


y=df_origin['y']
x=df_origin.drop(['y'],axis=1)
x=np.array(x)

# x=x.reshape(-1, 1)
# y=np.array(y)
# y=y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


for i in range(1,6):
    polynomial_features= PolynomialFeatures(degree=i)
    x_poly = polynomial_features.fit_transform(x)
    model = LinReg()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r_score = r2_score(y,y_poly_pred)
    print('Degree=',i,'  R-score',r_score)


# In[ ]:




