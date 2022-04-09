#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras


# In[2]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[3]:


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# In[4]:


x_train.shape,x_test.shape


# In[5]:


x_train = x_train/255.0
x_test = x_test/255.0


# In[6]:


nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples,nx*ny*nrgb))


# In[7]:


nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples,nx*ny*nrgb))


# In[8]:


from sklearn.neighbors import KNeighborsClassifier


# In[9]:


knn=KNeighborsClassifier(n_neighbors=7)


# In[10]:


knn.fit(x_train2,y_train)


# In[15]:


y_pred_knn=knn.predict(x_test2)
y_pred_knn


# In[20]:


accuracy_score(y_pred_knn,y_test)


# In[19]:


print(classification_report(y_pred_knn,y_test))


# In[22]:


confusion_matrix(y_pred_knn,y_test)


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sn
sn.heatmap(confusion_matrix, annot=True)
plt.xlabel('Y_Predicted')
plt.ylabel('Y_Truth')


# In[27]:


cm=confusion_matrix(y_pred_knn,y_test)


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True,fmt="d")
plt.xlabel('Y_Predicted')
plt.ylabel('Y_Truth')


# In[ ]:




