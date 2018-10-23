import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from modulesClass import subsample
from modulesClass import decisionTree, randomForest

treeData = pd.read_csv('covtype.csv')
print('Dataset shape: ', treeData.shape)
print('Tree Types: ', treeData.Cover_Type.unique())
classPct = treeData.groupby(treeData.Cover_Type).size()/treeData.shape[0]
print('Class percentage: \n', classPct)

'''
Our dataset is large, so we will subsample.
From the above percentages we keep the 
same percentages of each class in our subsample. 
'''
sampleSize = 20000
train, test = subsample(treeData, classPct, sampleSize)

trainX = train.loc[:, train.columns != 'Cover_Type']
trainY = train['Cover_Type'].astype(str)
testX = train.loc[:, train.columns != 'Cover_Type']
testY = train['Cover_Type'].astype(str)

train[train.dtypes[(train.dtypes == "float64") | (train.dtypes == "int64")].index.values].hist(figsize=[17, 17])
plt.tight_layout()
plt.show()
# From the above histograms we see that the range in values for the features varies
# so if we tried to run a distance based method the features with the largest range will dominate the others.
# We normalize our data.
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)

scaler.fit(testX)
testX = scaler.transform(testX)

#=============================================================
#=============================================================

# 1. Decision Tree Classifier
modelDT, mse, cf = decisionTree(trainX, trainY, testX, testY)
print('DecisionTree - Mean Square Error: ', mse)
print('DecisionTree - Confusion Matrix: \n', cf)

# 2. RandomForest
modelForest, mse, cf = randomForest(trainX, trainY, testX, testY)
print('RandomForest - Mean Square Error: ', mse)
print('RandomForest - Confusion Matrix: \n', cf)
