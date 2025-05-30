#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# In[33]:


df = pd.read_csv("C:\\Users\\karth\\OneDrive\\Desktop\\stock\\Tesla.csv")
df.head()


# In[34]:


df.shape


# In[35]:


df.describe()


# In[36]:


df.info()


# In[37]:


plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()


# In[38]:


df.head()


# In[39]:


df[df['Close'] == df['Adj Close']].shape


# In[40]:


df = df.drop(['Adj Close'], axis=1)


# In[41]:


df.isnull().sum()


# In[42]:


features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()


# In[43]:


plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()


# In[44]:


splitted = df['Date'].str.split('/', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

df.head()


# In[45]:


df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()


# In[46]:


data_grouped = df.drop('Date', axis=1).groupby('year').mean()
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()

# This code is modified by Susobhan Akhuli


# In[47]:


df.drop('Date', axis=1).groupby('is_quarter_end').mean()


# In[48]:


df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# In[49]:


plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()


# In[50]:


plt.figure(figsize=(10, 10)) 

# As our concern is with the highly 
# correlated features only so, we will visualize 
# our heatmap as per that criteria only. 
sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()

# This code is modified by Susobhan Akhuli


# In[51]:


features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


# In[52]:


get_ipython().system(' pip install xgboost')



# In[53]:


from xgboost import XGBClassifier


# In[54]:


models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()


# In[55]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()

# This code is modified by Susobhan Akhuli


# In[56]:


get_ipython().system(' pip install joblib')


# In[64]:


import pandas as pd

# Load unseen data
X_new = pd.DataFrame({
    "Open": [130.813739],
    "High": [133.182620],
    "Low":  [128.257229]
})

# Make sure to do the same preprocessing as X_train
# For example: scaling, encoding, dropping columns, etc.

# Then predict
y_pred = loaded_model.predict(X_new)
y_proba = loaded_model.predict_proba(X_new)[:, 1]

print("Predictions:", y_pred)
print("Probabilities:", y_proba)


# In[ ]:




