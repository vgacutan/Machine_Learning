import numpy as np
import RTLearner
from random import randint


class BagLearner(object):

    def __init__(self, learner=RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False):
            
        #create learners
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def author(self):
         return 'vgacutan3'
         
    def addEvidence(self, setX, setY):
        randomIx = []
        for i in range(int(0.6 * setX.shape[0])):
            randomIx.append(randint(0, setX.shape[0] - 1))
        trainSetX = []
        trainSetY = []
        for learner in self.learners:            
            for ix in randomIx:
                trainSetX.append(setX[ix])
                trainSetY.append(setY[ix])
            learner.addEvidence(np.array(trainSetX),np.array(trainSetY))

    def query(self, xt):     
        bagsCnt = self.learners
        predY = None
        for i in bagsCnt:
            if predY is None:
                predY = i.query(xt)
            else:
                predY = np.add(predY, i.query(xt))    
        return predY / len(self.learners)