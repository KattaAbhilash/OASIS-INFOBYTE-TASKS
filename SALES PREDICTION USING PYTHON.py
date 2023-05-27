#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  #linear algebra
import pandas as pd  #data processing


# Reading CSV File

# In[5]:


df=pd.read_csv('C:\\Users\\Administrator\\Desktop\\data\\Advertising.csv')  


# In[6]:


df.head()  #returns first 5 entries


# In[7]:


df.tail()  #returns last 5 entries


# In[8]:


#returns tuple of shape (Rows, columns) of dataframe
df.shape


# In[9]:


#prints information about the dataframe
df.info()


# In[10]:


#returns numerical description of the data in the dataframe
df.describe()


# Droping the Column

# In[11]:


#dropping the column 'Unnamed: 0'
df=df.drop(columns=["Unnamed: 0"])


# In[12]:


df  #return dataframe


# In[13]:


x=df.iloc[:, 0:-1]


# In[14]:


x


# In[15]:


y=df.iloc[:,-1]


# In[16]:


y


# Train Test Split

# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)


# In[19]:


x_train


# In[20]:


x_test


# In[21]:


y_train


# In[22]:


y_test


# In[23]:


x_train=x_train.astype(int)
y_train=y_train.astype(int)
x_test=x_test.astype(int)
y_test=y_test.astype(int)


# In[24]:


from sklearn.preprocessing import StandardScaler
Sc=StandardScaler()
x_train_scaled=Sc.fit_transform(x_train)
x_test_scaled=Sc.fit_transform(x_test)


# Applying Linear Regression

# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


lr=LinearRegression()


# In[27]:


lr.fit(x_train_scaled,y_train)


# In[28]:


y_pred=lr.predict(x_test_scaled)


# Evaluate the performance of a Linear Regerssion Model

# In[29]:


from sklearn.metrics import r2_score


# In[30]:


r2_score(y_test,y_pred)


# Analyzing Data By Scatter Plot

# In[31]:


import matplotlib.pyplot as plt


# In[32]:


plt.scatter(y_test,y_pred,c='g')


# In[ ]:




