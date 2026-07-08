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
import joblib
from pathlib import Path


# In[11]:


X = pd.read_csv('C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Heart attack and diseases model\\backend\\database\\disease_prediction _edited.csv')
Y = pd.read_csv('C:\\Users\\Ahmed Salah\\Desktop\\private MO\\programming\\projects\\ML-DL projects\\Heart attack and diseases model\\backend\\database\\result.csv')
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
X = scaler.fit_transform(X)


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


# In[14]:


model = LogisticRegression()
model.fit(x_train ,y_train)


# In[15]:


print(model.score(x_test , y_test))


# In[16]:


BASE_DIR = Path("Heart attack and diseases model").resolve().parent.parent
model_path = BASE_DIR / "backend" / "disease_model.joblib"
joblib.dump(model, model_path)
print(f"the model has saved in: {model_path}")
joblib.dump(scaler, BASE_DIR / "backend" / "scaler.joblib")

# In[ ]:




