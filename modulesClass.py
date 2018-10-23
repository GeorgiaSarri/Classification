import sklearn as sk
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#============================================================
# subsample:
# input:
#   data:       original dataset
#   classPct:   percentahe of each class in original dataset
#   sampleSize: the subsample's size
# description:
#   Breaks the original dataset into a train and test set
#   reducing the dataset's size while keeping the original
#   percentage of each class.
#============================================================
def subsample(data, classPct, sampleSize):
    # Find NAs
    cols = []
    for x in data.columns.values:
        if data[x].isnull().sum() > 0:
            cols.append(x)
    
    print('Number of columns with NAs: ', len(cols))
    
    train = pd.concat([
                        data.loc[data.Cover_Type == 1].sample(int(sampleSize*classPct[classPct.index==1])),
                        data.loc[data.Cover_Type == 2].sample(int(sampleSize*classPct[classPct.index==2])),
                        data.loc[data.Cover_Type == 3].sample(int(sampleSize*classPct[classPct.index==3])),
                        data.loc[data.Cover_Type == 4].sample(int(sampleSize*classPct[classPct.index==4])),
                        data.loc[data.Cover_Type == 5].sample(int(sampleSize*classPct[classPct.index==5])),
                        data.loc[data.Cover_Type == 6].sample(int(sampleSize*classPct[classPct.index==6])),
                        data.loc[data.Cover_Type == 7].sample(int(sampleSize*classPct[classPct.index==7]))
                        ])
    test = pd.concat([
                        data.loc[data.Cover_Type == 1].sample(int(sampleSize*classPct[classPct.index==1])),
                        data.loc[data.Cover_Type == 2].sample(int(sampleSize*classPct[classPct.index==2])),
                        data.loc[data.Cover_Type == 3].sample(int(sampleSize*classPct[classPct.index==3])),
                        data.loc[data.Cover_Type == 4].sample(int(sampleSize*classPct[classPct.index==4])),
                        data.loc[data.Cover_Type == 5].sample(int(sampleSize*classPct[classPct.index==5])),
                        data.loc[data.Cover_Type == 6].sample(int(sampleSize*classPct[classPct.index==6])),
                        data.loc[data.Cover_Type == 7].sample(int(sampleSize*classPct[classPct.index==7]))
                        ])
    return train, test

#=====================================
# Classification Modules:
# 1. Decision Tree
# 2. Random Forest
#=====================================

def decisionTree(trainX, trainY, testX, testY, cv=5):
    parameters = [
        {'max_depth': list(range(1, 15))},
    ]

    clf_tree = GridSearchCV(DecisionTreeClassifier(), parameters, cv=cv)
    clf_tree = clf_tree.fit(trainX, trainY)
    Y_pred = clf_tree.predict(testX)

    return clf_tree, mean_squared_error(testY, Y_pred), confusion_matrix(testY, Y_pred, labels=['1','2','3','4','5','6','7'])

#2. RandomForest
def randomForest(trainX, trainY, testX, testY, cv=5, n_jobs = -1):
    parameters = [
        {'max_depth': list(range(1, 15)),
         'n_estimators': list(range(1, 50))},
    ]

    clf_forest = GridSearchCV(RandomForestClassifier(), parameters, cv=cv, n_jobs=n_jobs)
    scores = cross_val_score(clf_forest, trainX, trainY)
    print('RandomForest - Mean of scores: ', scores.mean())

    clf_forest = clf_forest.fit(trainX, trainY)

    Y_pred = clf_forest.predict(testX)

    return clf_forest, mean_squared_error(testY, Y_pred), confusion_matrix(testY, Y_pred, labels=['1','2','3','4','5','6','7'])

