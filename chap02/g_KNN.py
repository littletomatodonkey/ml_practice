import matplotlib

from numpy import *
import operator

from pandas import DataFrame,Series


import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 0.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classifyKNN(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDists = sqDiffMat.sum(axis=1)
    dists = sqDists ** 0.5
    sortedDistIndicies = dists.argsort()
    classCnt = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCnt[voteILabel] = classCnt.get(voteILabel, 0) + 1
    sortedClassCnt = sorted(classCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCnt[0][0]


def KNN_test():
    group, labels = createDataSet()
    resType = classifyKNN([0, 0], group, labels, 3)
    print(resType)

# read file and convert it to mat data
def file2Mat(fn):
    fr = open(fn)
    arrLines = fr.readlines()
    numberOfLines = len(arrLines)
    result = zeros((numberOfLines, 3))
    labelVector = []
    index = 0
    for line in arrLines:
        line = line.strip()
        listFromLine = line.split('\t')
        result[index, :] = listFromLine[0:3]  # just 3 numbers are included
        labelVector.append(int(listFromLine[-1]))
        index += 1
    return result, labelVector


# plot data without data
def plotWithSingleColor():
    dataDatingMat, datingLabels = file2Mat('./data/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter( dataDatingMat[:, 1], dataDatingMat[:, 2] )
    plt.show()
    plt.savefig('single_color.png')

# plot data with color
def plotWithScatter():
    dataDatingMat, datingLabels = file2Mat('./data/datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter( dataDatingMat[:, 1], dataDatingMat[:, 2], 
               15*array(datingLabels), 15*array(datingLabels))
    plt.show()
# use the following to imort data :
#    dataDatingMat, datingLabels = file2Mat('./data/datingTestSet2.txt')
def autoNorm(dataSet):
    minV = dataSet.min(0)
    maxV = dataSet.max(0)
    ranges = maxV - minV
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minV, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minV

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2Mat('./data/datingTestSet2.txt')
    normMat, ranges, minV = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTest = int(m * hoRatio)
    errorCnt = 0.0
    for i in range(numTest):
        res = classifyKNN(normMat[i,:], normMat[numTest:m, :], datingLabels[numTest:m], 3)
        print "the classifier came back with %d, the real number is : %d" \
                    % (res, datingLabels[i] )
        if( res != datingLabels[i] ):
            errorCnt += 1
    print "the total error rate is %f" % (errorCnt/ float(numTest))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
