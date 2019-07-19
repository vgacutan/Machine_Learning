# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 23:42:19 2017

@author: VMG
"""

import numpy as np
import math

class KNNLearner():
    def __init__(self, k):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def query(self, Xtest):
        Y = np.zeros((Xtest.shape[0], 1), dtype='float')

        for i in range(Xtest.shape[0]):
            dist = (self.Xtrain[:, 0] - Xtest[i, 0])**2 + (self.Xtrain[:, 1] - Xtest[i, 1])**2
            knn = [self.Ytrain[knni] for knni in np.argsort(dist)[:self.k]]
            Y[i] = np.mean(knn)

        return Y


if __name__ == '__main__':
    data = np.genfromtxt('Data/ripple.csv', delimiter=',')
    train_len = int(data.shape[0] * .6)

    train_data = data[:train_len, :]
    test_data = data[train_len:, :]

    Xtrain = train_data[:, :2]
    Ytrain = train_data[:, 2]
    Xtest = test_data[:, :2]
    Ytest = test_data[:, 2]

    learner = KNNLearner(k=2)
    learner.addEvidence(Xtrain, Ytrain)

    print Ytrain

    Y = learner.query(Xtrain)
    predY = [float(x) for x in Y]
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    print
    print 'KNN'
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0, 1]

    Y = learner.query(Xtest) # get the predictions
    predY = [float(x) for x in Y]
    rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    print "corr: ", c[0, 1]