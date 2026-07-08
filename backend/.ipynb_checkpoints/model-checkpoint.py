#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn


# In[11]:


X = pd.read_csv('disease_prediction _edited.csv')
Y = pd.read_csv('result.csv')
""" male = 1 
    female = 0
    yes = 1
    no = 0
    physical_activiy in the dataset
          low = 0
          medium = 1
          high = 2
"""
scaler = StandardScaler()
numerical_features = [
    "age",
    "glucose_mg_dl",
    "cholesterol_mg_dl",
    "systolic_bp",
    "diastolic_bp",
    "bmi",
    "heart_rate",
    "MAP",
    "RPP Rate Pressure Product",
    "PP Pulse Pressure",
    "Unhealthy Lifestyle Score",
    "Atherogenic Index Coefficient",
    "Smoking-Hypertension Interaction",
    "Cardiac Adiposity Proxy",
    "Cardiovascular Stress Index"
]
X[numerical_features] = scaler.fit_transform(X[numerical_features])


# In[12]:


x_train ,x_temp ,y_train ,y_temp = train_test_split(
    X ,Y,
    test_size=0.3,
    random_state=42,
)
x_val ,x_test ,y_val ,y_test = train_test_split(
    x_temp ,y_temp,
    test_size=0.5,
    random_state=42,
)


# In[13]:


print(x_train.shape ,y_train.shape)
print(x_val.shape ,y_val.shape)
print(x_test.shape ,y_test.shape)


# In[18]:


model = LogisticRegression()
model.fit(x_train ,y_train)


# In[19]:


print(model.score(x_test , y_test))


# In[ ]:





# In[ ]:




