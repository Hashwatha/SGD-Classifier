#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
iris=load_iris()


# In[36]:


df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())


# In[37]:


x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)


# In[38]:


accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")


# In[39]:


cm=confusion_matrix(y_test,y_pred)
print("Confufion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print("Name:Hashwatha M")
print("Reg no:212223240051")


# In[ ]:




