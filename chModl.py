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