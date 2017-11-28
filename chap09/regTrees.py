# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 14:23:19 2017

@author: IBM
"""
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet( dataSet,  feature, value ):
    mat0 = dataSet[ nonzero(dataSet[:, feature] >  value)[0], : ]
    mat1 = dataSet[ nonzero(dataSet[:, feature] <= value)[0], : ]
    return mat0, mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit( dataSet, leafType, errType, ops )
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet( dataSet, feat, val )

    retTree['left']  = createTree( lSet, leafType, errType, ops )
    retTree['right'] = createTree( rSet, leafType, errType, ops )
    return retTree

def chooseBestSplit( dataSet, leafType=regLeaf, errType=regErr, ops=(1,4) ):
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有值相等则退出
    if len(dataSet[:,-1].T.tolist()[0]) == 1:
        return None, leafType(dataSet)
    m,n = shape( dataSet )
    S = errType( dataSet )
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in dataSet[:,featIndex].T.tolist()[0]:
            mat0, mat1 = binSplitDataSet( dataSet, featIndex, splitVal)
            if ( shape(mat0)[0] < tolN ) or ( shape(mat1)[0] < tolN ):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS     = newS
    # 如果误差减少不大则退出
    if ( S - bestS ) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet( dataSet, bestIndex, bestValue )
    # 如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN ) or ( shape(mat1)[0] < tolN ):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def isTree( obj ):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean( tree['left'] )
    if isTree(tree['right']):
        tree['right'] = getMean( tree['right'] )
    return ( tree['left'] + tree['right'] ) / 2

# 回归数减枝函数（后减枝）
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree

def testPrune():
    myDat2 = loadDataSet( 'ex2.txt' )
    myMat2 = mat( myDat2 )
    myTree = createTree( myMat2, ops = (0,1) )
    print 'tree without prune...'
    print myTree
    myDataSet = loadDataSet('ex2test.txt')
    myMat2Test = mat( myDataSet )
    newTree = prune( myTree, myMat2Test )
    return myTree, newTree

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

def testModelForest():
    trainMat = mat( loadDataSet('bikeSpeedVsIq_train.txt') )
    testMat  = mat( loadDataSet('bikeSpeedVsIq_test.txt') )
    myTree = createTree( trainMat, ops = (1,20) )
    yHat = createForeCast( myTree, testMat[:,0] )
    regCorr = corrcoef(yHat, testMat[:,1], rowvar=0)[0, 1]
    print 'regCorr : ', regCorr

    myTree = createTree( trainMat, modelLeaf, modelErr, (1,20) )
    yHat = createForeCast( myTree, testMat[:,0], modelTreeEval)
    modelRegCorr = corrcoef(yHat, testMat[:,1], rowvar=0)[0, 1]
    print 'modelRegCorr : ', modelRegCorr