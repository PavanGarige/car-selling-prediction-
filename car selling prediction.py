#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np


# In[114]:


import os
data=pd.read_csv('C:\\Users\\Mahishekar\\Documents\\car data.csv')


# In[115]:


data


# In[116]:


data.describe()


# In[117]:


data['Selling_Price'].isnull().sum()


# In[118]:


print(data['Selling_Price'].unique().values())


# In[119]:


data['current_year']=2020


# In[120]:


#data['current year']


# In[121]:


data['no_yrs']=data['current_year']-data['Year']


# In[122]:


data.columns


# In[123]:


data=data[[ 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner',
       'no_yrs']]


# In[124]:


data.head()


# In[ ]:





# In[125]:


#data.columns


# In[126]:


#data=data[[]]


# In[127]:


data=pd.get_dummies(data,drop_first=True)


# In[128]:


data.head()


# In[129]:


data.corr()


# In[130]:


import seaborn as sns
sns.pairplot(data)


# In[131]:


import matplotlib.pyplot as plt


# In[133]:


#heat=data.corr()
#corre=heat.index
plt.figure(figsize=(18,17))
h=sns.heatmap(data.corr(),annot=True)


# In[135]:


X=data.iloc[:,1:]
Y=data.iloc[:,0]


# In[136]:


Y


# In[139]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()


# In[140]:


model.fit(X,Y)


# In[141]:


model.feature_importances_


# In[142]:


aa=pd.Series(model.feature_importances_,index=X.columns)


# In[143]:


aa.nlargest(5).plot(kind='barh')
plt.show()


# In[ ]:




