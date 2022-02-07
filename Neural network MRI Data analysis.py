#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading the csv file and importing the packages

import tensorflow as tf
import keras
import pandas as pd
import matplotlib
import sklearn


# In[44]:


df = pd.read_csv('C:/Users/vd21aaf/Downloads/V Dass ML project MRI data adjusted.csv')


# In[45]:


df


# In[46]:


dataset = df.values


# In[47]:


#converting the pandas dataframe into array
dataset


# In[48]:


# splitting the dataset into our input features and the label we wish to predict
X = dataset[:,0:28]
Y = dataset[:,28]


# In[49]:


# Data Normalization
# scaling the data to be betweeen 0 and 1
from sklearn import preprocessing


# In[50]:


min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)


# In[51]:


X_scale


# In[ ]:


# Splitting the dataset into test and validation set


# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)


# In[54]:


X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)


# In[55]:


print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


# In[ ]:


# Building and Training Our First Neural Network
# Using Keras to build the neural network architecture


# In[56]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# Using TensorFlow backend.
# We will be using the Sequential model, which means that we merely need to describe the layers above in sequence. 
# Our neural network has three layers:
# Hidden layer 1: 30 neurons, ReLU activation
# Hidden layer 2: 30 neurons, ReLU activation
# Output Layer: 1 neuron, Sigmoid activation


# In[57]:


model = Sequential([
    Dense(32, activation='relu', input_shape=(28,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])


# In[ ]:


# Before we start our training, we have to configure the model by

# Telling it what algorithm you want to use to do the optimization (we'll use stochastic gradient descent)
# Telling it what loss function to use (for binary classification, we will use binary cross entropy)
# Telling it what other metrics you want to track apart from the loss function (we want to track accuracy as well)


# In[58]:


model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# The function is called 'fit' as we are fitting the parameters to the data. We specify:

what data we are training on, which is X_train and Y_train
the size of our mini-batch
how long we want to train it for (epochs)
what our validation data is so that the model will tell us how we are doing on the validation data at each point.


# In[59]:


hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))


# In[60]:


model.evaluate(X_test, Y_test)[1]


# In[61]:


import matplotlib.pyplot as plt


# In[62]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[64]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


# Train a model which will overfit, Model2


# In[65]:


model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(28,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))


# In[66]:


plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[67]:


plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


# To address the overfitting we see in Model 2, we'll incorporate L2 regularization and dropout in our third model here 
# (Model 3)


# In[68]:


from keras.layers import Dropout
from keras import regularizers


# In[69]:


model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(28,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])


# In[70]:


model_3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_3 = model_3.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))


# In[71]:


plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()


# In[72]:


plt.plot(hist_3.history['accuracy'])
plt.plot(hist_3.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:




