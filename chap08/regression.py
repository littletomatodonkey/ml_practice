# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 09:38:43 2017

@author: IBM
"""
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def standRegres(xArr, yArr, plotCurve = True):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det( xTx ) == 0:
        print "this matrix is singular, can not do inverse"
        return
    ws = xTx.I * ( xMat.T * yMat )
    if plotCurve:
        yHat = xMat * ws
        fig = plt.figure()
        ax = fig.add_subplot( 111 )
        ax.scatter( xMat[:,1].flatten().A[0], yMat[:,0].flatten().A[0] )
        xCopy = xMat.copy()
        xCopy.sort( 0 )
        yHat = xCopy * ws
        ax.plot( xCopy[:,1], yHat )
        plt.show()
    return ws

def lwlr( testPoint, xArr, yArr, k = 1.0 ):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape( xMat )[0]
    weights = mat(eye((m)))
    for j in range( m ):
        diffMat = testPoint - xMat[j, :]
        weights[j,j] = exp( diffMat*diffMat.T / (-2.0*k**2) )
    xTx = xMat.T * weights * xMat
    if linalg.det( xTx ) == 0:
        print "this matrix is singular, can not do inverse"
        return
    ws = xTx.I * ( xMat.T * (weights * yMat) )
    return testPoint * ws

def lwlrTest( testArr, xArr, yArr, k=1.0, plotCurve = True ):
    m = shape( testArr )[0]
    yHat = zeros( m )
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)

    if plotCurve:
        xMat = mat(xArr)
        srtInd = xMat[:, 1].argsort(0)
        xSort = xMat[srtInd][:,0,:]
        fig = plt.figure()
        ax = fig.add_subplot( 111 )
        ax.plot( xSort[:,1], yHat[srtInd] )
        ax.scatter( xMat[:,1].flatten().A[0], mat(yArr).T[:,0].flatten().A[0], s=2.0, c='red' )
        plt.show()
    return yHat

def rssError( yArr, yHatArr ):
    return ( (yArr-yHatArr)**2 ).sum()

def ridgeRegres( xMat, yMat, lam = 0.2 ):
    xTx = xMat.T * xMat
    denom = xTx + eye( shape(xMat)[1] ) * lam
    if linalg.det( denom ) == 0:
        print "this matrix is singular, can not do inverse"
        return
    ws = denom.I * ( xMat.T * yMat )
    return ws

def ridgeTest( xArr, yArr ):
    xMat = mat( xArr )
    yMat = mat( yArr ).T
    yMat = yMat - mean( yMat )
    xMat = ( xMat - mean(xMat,0) ) / var( xMat, 0 )
    numTestPts = 30
    wMat = zeros( ( numTestPts, shape(xMat)[1] ) )
    for i in range(numTestPts):
        ws = ridgeRegres( xMat, yMat, exp(i-10) )
        wMat[i,:] = ws.T
    return wMat

def testRidgeRegres():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest( abX, abY )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot( ridgeWeights )
    plt.show()

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# 前向逐步线性回归
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMat = yMat - mean( yMat, 0 )
    xMat = regularize( xMat )
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros( (n,1) )
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError( yMat.A, yTest.A )
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

from time import sleep
import json
import urllib2
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)