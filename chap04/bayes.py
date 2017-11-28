# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 13:51:21 2017

@author: IBM
"""
from numpy import *
import sys # 判断输入参数的个数
import re  # 正则表达式

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 从数据集中创建列表
def createVocabList(dataSet):
    #return {}.fromkeys(sum(dataSet, [])).keys()
    # 下面的是书籍中给的方法，上面的方法比较简洁
    vocabSet = set([])
    for doc in dataSet:
        vocabSet |= set(doc)
    return list(vocabSet)

# 判断inputSet中是否存在vocabList
# 返回列表，如果存在，对应值为1，否则为0
def setOfWords2vec( vocabList, inputSet ):
    print 'enter, len is : '
    returnVec = [0] * len(vocabList)
    for s in inputSet:
        if s in vocabList:
            returnVec[vocabList.index( s )] = 1
        else:
            print( "the word : %s is not in my vocabulary!" % s )
    return returnVec

# 保存单词出现的个数
def bagOfWords2vec( vocabList, inputSet ):
    returnVec = [0] * len(vocabList)
    for s in inputSet:
        if s in vocabList:
            returnVec[vocabList.index( s )] += 1
        else:
            print( "the word : %s is not in my vocabulary!" % s )
    return returnVec    

def trainNB0( trainMat, trainCategory ):
    numTrainDocs = len( trainMat )
    numWords = len( trainMat[0] )
    pAbusive = sum( trainCategory ) / float( numTrainDocs )
    # 为了防止因为没有出现该词汇而导致概率为0的情况，在此将初始值增加1
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if( trainCategory[i] == 1 ):
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    #p1Vec = p1Num / p1Denom
    #p0Vec = p0Num / p0Denom
    # 因为概率太小，之后相乘时可能会导致数据舍入误差等，在此取对数，同时也不影响数据的变化趋势
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive

# 测试bayes分类的效果，返回值是P(W/A), P(NW/A), P(NA)  N表示取反
def testNBayes():
    listPost, listClasses = loadDataSet()
    myVocabList = createVocabList( listPost )
    trainMat = []
    for postDoc in listPost:
        trainMat.append( setOfWords2vec( myVocabList, postDoc ) )
    p0V, p1V, pAb = trainNB0( trainMat, listClasses )
    print p0V, p1V, pAb
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array( setOfWords2vec(myVocabList, testEntry) )
    print testEntry, 'classified as : ', classifyNB(thisDoc, p1V, p0V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array( setOfWords2vec(myVocabList, testEntry) )
    print testEntry, 'classified as : ', classifyNB(thisDoc, p1V, p0V, pAb)
    return p0V, p1V, pAb

def classifyNB( vec2Classify, p1Vec, p0Vec, pClass1 ):
    p1 = sum( vec2Classify * p1Vec ) + log( pClass1 )  # 因为前文中，所有概率均取了对数，在此也将乘法变成了对数
    p0 = sum( vec2Classify * p0Vec ) + log( 1 - pClass1 )
    if p1 > p0:
        return 1
    else:
        return 0
#分割字符串
def splitText(str = 'A drunken soldier attempts to rape a fisherman\'s wife during the evacuation.'):
    pureSplit = str.split()
    print 'puresplit : ',  pureSplit

    regEx = re.compile( '\\W*' )
    regSplit = regEx.split( str )
    print 'regSplit : ' , regSplit

    removeSpaceSplit = [tok for tok in regSplit if len(tok) > 0]
    print 'removeSpaceSplit : ', removeSpaceSplit
    return removeSpaceSplit

# 提取字符串中的单词
def textParse(bigStr):
    import re
    listOfTokens = re.split( r'\W*', bigStr )
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse( open('email/spam/%d.txt' % i ).read() )
        docList.append( wordList )
        fullText.extend( wordList )
        classList.append( 1 )
        

        wordList = textParse( open('email/ham/%d.txt' % i ).read() )
        docList.append( wordList )
        fullText.extend( wordList )
        classList.append( 0 )

    vocabList = createVocabList( docList )
    
#    print( docList )
    # 从50个数据集中选取10个作为测试集，剩下40个作为训练
    trainingSet = range(50); testSet = []
    for i in range(10):
        rndIndex = int( random.uniform(0, len( trainingSet )) )
        testSet.append( trainingSet[rndIndex] )
        del( trainingSet[rndIndex] )
    # 0,1矩阵的训练矩阵
    trainingMat = []
    trainClasses = []
    
    for docIndex in trainingSet:
        trainingMat.append( bagOfWords2vec( vocabList, docList[docIndex] ) ) # 表示vocabList中的单词在docList[docIndex]中是否存在
        trainClasses.append( classList[docIndex] )  # docList[docIndex]这个列表是否为垃圾邮件
    p0V, p1V, pSpam = trainNB0( trainingMat, trainClasses )


    errorCnt = 0
    for docIndex in testSet:
        wordVector = bagOfWords2vec( vocabList, docList[docIndex] )
        if classifyNB(wordVector, p1V, p0V, pSpam) != classList[docIndex]:
            errorCnt += 1
    print 'the error rate is : ', float(errorCnt)/len(testSet)



