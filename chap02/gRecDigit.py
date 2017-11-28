# -*- coding:utf-8 -*-

from numpy import *
from os import listdir
import g_KNN

# 将存储数字形状信息的文本文件转化为一个向量
def img2vec(fn):
    retvec = zeros((1, 1024))
    fr = open(fn)
    for i in range(32):
        lstr = fr.readline()
        for j in range(32):
            retvec[0, 32*i+j] = int(lstr[j])
    return retvec


# result: the total number of errors is: 11
# the total error rate is: 0.011628
def handwritingClassTest():
    # training collection
    trainingDir = './digits/trainingDigits'
    testDir = './digits/testDigits'
    hwLabels = []
    trainingFl = listdir(trainingDir)
    m = len( trainingFl )
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fnStr = trainingFl[i]
        fileStr = fnStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vec(trainingDir + '/%s' % fnStr)
    
    # test collection 
    testFl = listdir(testDir)
    errorCnt = 0
    mTest = len(testFl)
    for i in range(mTest):
        fnStr = testFl[i]
        fileStr = fnStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vecUnderTest = img2vec(testDir + '/%s' % fnStr)
        classRes = g_KNN.classifyKNN(vecUnderTest, trainingMat, hwLabels, 3)
        
        if( classRes != classNum ):
            errorCnt += 1
    print('\nthe total number of errors is: %d' % errorCnt)
    print('\nthe total error rate is: %f' % (errorCnt/float(mTest)))
        
        
        
        
        
        
        
        
        
        
        
        


