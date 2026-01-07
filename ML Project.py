#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[5]:


df = pd.read_csv(r"C:\Users\Hp\Desktop\datasets\diabetes.csv")   # downloaded from Kaggle
print(df.head())


# In[14]:


columns_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[columns_with_missing] = df[columns_with_missing].replace(0, np.nan)
print(df)


# In[15]:


for col in columns_with_missing:
    df[col].fillna(df[col].median(), inplace=True)
print(df)


# In[16]:


def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

for col in columns_with_missing:
    df = remove_outliers(df, col)
print(df)


# In[17]:


scaler = StandardScaler()

features = df.drop("Outcome", axis=1)
target = df["Outcome"]

features_scaled = scaler.fit_transform(features)
print(df)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)
print(df)


# In[11]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[13]:


plt.boxplot(df["Glucose"])
plt.title("Glucose Level Distribution")
plt.show()


# In[ ]:




