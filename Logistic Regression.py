#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
digits = load_digits()


# In[19]:


print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)


# In[20]:


digits.data.shape


# In[21]:


digits.target.shape


# In[22]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
for index, (image, label) in enumerate (zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 30)


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state = 2)


# In[26]:


print(X_train.shape)


# In[27]:


print(y_train.shape)


# In[28]:


print(X_test.shape)


# In[29]:


print(y_test.shape)


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)


# In[33]:


print(logisticRegr.predict(X_test[0].reshape(1,-1)))


# In[34]:


print(logisticRegr.predict(X_test[0:10]))


# In[35]:


predictions = logisticRegr.predict(x_test)


# In[36]:


score = logisticRegr.score(X_test, y_test)
print(score)


# In[38]:


cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
# sum of all number in matrix is equal to total number of observation
# accuracy is determine by numbers in diagonal


# In[39]:


X_test.shape


# In[ ]:


# creating heatmap for confusion matrix


# In[41]:


plt.figure(figsize= (9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidth=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)


# In[51]:


index = 0
classifiedIndex = []
for predict, actual in zip(predictions, y_test):
    if predict==actual:
        classifiedIndex.append(index)
    index +=1
plt.figure(figsize=(30,10))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1, 4, plotIndex +1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}" .format(predictions[wrong], y_test[wrong]), fontsize=30)


# In[ ]:




