
import numpy as np
from random import randint


class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
   
    def author(self):
        return 'vgacutan3'

    def buildtree(self, trainSetX, trainSetY):    
        if len(np.unique(trainSetY)) == 1:
            return np.array([-1, trainSetY[0], -1, -1])          
        if trainSetX.shape[0] == 0:
            return np.zeros(trainSetX.shape[0])
        if trainSetX.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(trainSetY), -1, -1])       
        # splitting random values
        def splitbrnch(trainSetX, rowcnt = trainSetX.shape[0]):
            xtidx = randint(0, trainSetX.shape[1] - 1)
            rowcnt1 = randint(0, trainSetX.shape[0] - 1)
            rowcnt2 = randint(0, trainSetX.shape[0] - 1)
            randomSplit = (trainSetX[rowcnt1][xtidx] + trainSetX[rowcnt2][xtidx])/2
            RtreeVal = [x for x in range(trainSetX.shape[0]) if trainSetX[x][xtidx] > randomSplit]
            LtreeVal = [x for x in range(trainSetX.shape[0]) if trainSetX[x][xtidx] <= randomSplit]   
            return LtreeVal, RtreeVal, xtidx, randomSplit            
        LtreeVal, RtreeVal, xtidx, randomSplit = splitbrnch(trainSetX, trainSetX.shape[0])

        while (len(LtreeVal) < 1 or len(RtreeVal)) < 1:
            LtreeVal, RtreeVal, xtidx, randomSplit = splitbrnch(trainSetX, trainSetX.shape[0])
        Ltree = self.buildtree(np.array([trainSetX[x] for x in LtreeVal] ), np.array([trainSetY[x] for x in LtreeVal]))
        Rtree = self.buildtree(np.array([trainSetX[x] for x in RtreeVal]), np.array([trainSetY[x] for x in RtreeVal]))
        appendTree = np.vstack((Ltree, Rtree))
        if len(Ltree.shape) == 1:
            LtreeIdx = 2
        else:
            LtreeIdx = Ltree.shape[0] + 1
        Treeroot = [xtidx, randomSplit, 1, LtreeIdx]
        return np.vstack((Treeroot,appendTree))
        
    def addEvidence(self, setX, setY):
       self.rt = self.buildtree(setX, setY)
       
    def getPredictY(self, xt,rwIx=0):
        if int(self.rt[rwIx][0]) == -1:
            return self.rt[rwIx][1]
        if xt[int(self.rt[rwIx][0])] <= self.rt[rwIx][1]:
            return self.getPredictY(xt, rwIx + int(self.rt[rwIx][2]))
        else:
            return self.getPredictY(xt, rwIx + int(self.rt[rwIx][3]))
              
    def query(self, xt):
        y = []
        i = 0         
        for i in xt:
            y.append(self.getPredictY(i))      
        return np.array(y)
        
   