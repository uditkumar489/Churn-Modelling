# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 07:59:38 2018

@author: Udit
"""
# Artificial Neural Network

# IMPORTANT
# Please intall : {1.Theano  ,  2.Tensorflow  ,  3.Keras}  before executing it.


"""
Data PreProcessing
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The dataset 
dataset = pd.read_csv('Churn_Dataset.csv')
X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values

# Categorical data Encoded
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



"""
Making Artificial Neural Network
"""
# Keras libraries and packages imported
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() #initialising

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #input layer + first hidden layer

# Adding the second hidden layer (IMPROVEMENT)
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) #output layer

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #compiling

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) #fitting to dataset



"""
Making Predictions and Testing
"""
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
