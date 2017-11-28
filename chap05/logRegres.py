# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:32:39 2017

@author: IBM
"""
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open( 'testSet.txt' )
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append( [1.0, float(lineArr[0]), float(lineArr[1])] )
        labelMat.append( float( lineArr[2] ) )
    return dataMat, labelMat

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat( dataMatIn )
    label = mat( classLabels ).transpose()
    m, n = shape( dataMatrix )
    alpha = 0.001
    maxIter = 500
    weights = ones((n, 1))
    for k in range( maxIter ):
        h = sigmoid( dataMatrix * weights )
        error = ( label - h )
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()

def plotBestFit(wei):
    import matplotlib.pyplot as plt
#    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array( dataMat )
    n = shape( dataArr )[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int( labelMat[i] ) == 1:
            xcord1.append( dataArr[i, 1] ); ycord1.append( dataArr[i, 2] )
        else:
            xcord2.append( dataArr[i, 1] ); ycord2.append( dataArr[i, 2] )
    fig = plt.figure()
    ax = fig.add_subplot( 111 )
    ax.scatter( xcord1, ycord1, s = 30, c = 'red', marker = 's' )
    ax.scatter( xcord2, ycord2, s = 30, c = 'green' )
    x = arange( -3.0, 3.0, 0.1 )
    y = -( wei[0] + wei[1] * x ) / wei[2]

    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('Y1'); 
    plt.show()

# 可视化数据集以及最大梯度方法得到的分类结果
def visualziePlot(t = 1):
    dataArr, labelMat = loadDataSet()
    if t == 1:
        weights = gradAscent( dataArr, labelMat )
    elif t == 2:
        weights = stocGradAscent( array( dataArr), labelMat )
    plotBestFit( weights )

# 改进的随机梯度上升法，是一个在线的优化方法
# 对于任何添加的数据都可以通过训练数据，修正回归系数
def stocGradAscent(dataMat, classLabels, numIter = 150):
    m, n = shape( dataMat )
    weights = ones(n)
    for j in range( numIter ):
        dataIndex = range(m)
        for i in range( m ):
            alpha = 4 / (1.0+j+i) + 0.01
            rndIndex = int( random.uniform( 0, len(dataIndex) ) )
            h = sigmoid( sum( dataMat[rndIndex] * weights ) )
            error = classLabels[rndIndex] - h
            weights = weights + alpha * error * dataMat[rndIndex]
            del( dataIndex[rndIndex] )
    return weights

def classifyVector(inx, weights):
    
    prob = sigmoid( sum( inx*weights ) )
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    traingingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append( float(currLine[i]) )
        traingingSet.append( lineArr )
        trainingLabels.append( float(currLine[21]) )
    trainWeights = stocGradAscent(array(traingingSet), trainingLabels, 500)
    errorCnt = 0; numTestVec  =0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append( float( currLine[i] ) )
        if int( classifyVector(array(lineArr), trainWeights )) != int(currLine[21]):
            errorCnt += 1.0
    errorRate = errorCnt / numTestVec
    print 'the error rate of the test is : %f' % errorRate
    return errorRate

def mulitTest():
    numTest = 10; errorSum = 0.0
    for k in range( numTest ):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is : %f' % (numTest, errorSum/numTest)
