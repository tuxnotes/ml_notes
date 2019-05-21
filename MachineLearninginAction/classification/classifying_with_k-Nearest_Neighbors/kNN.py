import numpy as np
import operator

def createDataSet():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    inX: the input vector to classify
    dataSet: full matrix of training examples
    labels: a vector of labels
    k: the number of nearest neighbors to use in the voting
    """
    dataSetSize = dataSet.shape[0]  # get rows number of dataSet
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # diff matrix 
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  
    distances = sqDistances**0.5  # distance calculation
    sortedDistIndicies = distances.argsort()  # sort
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                             key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
