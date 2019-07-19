# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 23:39:48 2017

@author: VMG
"""

import matplotlib.pyplot as plt
import pandas as pd
#import KNNLearner as knn
import RTLearner as rt
import BagLearner as bl
import numpy as np


df = pd.DataFrame()

#test = [1, 2, 3, 4]
#test1 = [5, 6, 7, 8]

#df['test'] = test
#df['test1'] = test1

#df = df.set_index('test')

#print df

data = np.genfromtxt('Data/winequality-red.csv', delimiter=',')
train_len = int(data.shape[0] * .6)

train_data = data[:train_len, :]
test_data = data[train_len:, :]

Xtrain = train_data[:, :2]
Ytrain = train_data[:, 2]
Xtest = test_data[:, :2]
Ytest = test_data[:, 2]

indexes = []
#test_rmses = []
#train_rmses = []
test_cs = []
train_cs = []

print
print("Plotting RMSE for vatious bags... ")

for i in range(1, 100):

    #learner = knn.KNNLearner(k=i)
    learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": i}, bags=6, boost=False, verbose=False)
    learner.addEvidence(Xtrain, Ytrain)
    print(learner.author())
    
    Y = learner.query(Xtrain)
    predY = [float(x) for x in Y]
    #rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    #train_rmses.append(rmse)
    #print
    #print 'KNN'
    #print "In sample results"
    #print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    train_cs.append(c[0,1])
    #print "corr: ", c[0, 1]

    Y = learner.query(Xtest) # get the predictions
    predY = [float(x) for x in Y]
    #rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    #print
    #print "Out of sample results"
    #print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    #print "corr: ", c[0, 1]

    indexes.append(i)
    #test_rmses.append(rmse)
    test_cs.append(c[0,1])

#print test_cs
#print train_cs
df['Bags'] = indexes
#df['Out of sample RMSE'] = test_rmses
#df['In sample RMSE'] = train_rmses
df['Out of sample corr'] = test_cs
df['In sample corr'] = train_cs

df = df.set_index('Bags')

ax = df.plot(title='Bags vs corr - ripple.csv')
ax.set_xlabel('Bags')
ax.set_ylabel('RMSE')
plt.show()
print("Done.", "")