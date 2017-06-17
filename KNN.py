#coding=utf-8
import numpy as np
import operator
from os import listdir
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #获取第一维数,行的数目
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet #把输入inX点复制到整个dataSetSize的矩阵，计算出差值
    # tile(A,reps) construct an array by repeating A the number of times given by reps.
    #reps的数字从后往前分别对应A的第N个维度的重复次数
    #tile([1,2],(2,3)) ==>> [[1,2,1,2,1,2],[1,2,1,2,1,2]]
    sqDiffMat = diffMat**2
    sqDistances = np.sum(sqDiffMat,axis=1)
    distances = sqDistances**0.5 #算出inX与每个点的欧氏距离
    sortedDistIndicies = np.argsort(distances) #返回数值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #筛选出由近到远的label
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #返回voteIlabel里给定键的值，如果没有的键就返回0
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) #{}.iteritems 返回元祖列表的迭代器，并按元祖的第二个元素的次序进行降序排列
    return sortedClassCount[0][0]#返回出现次数最多的label

#KNN进行约会配对
#转化文本为numpy.array
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines() # 是一个list
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#数据归一化
def autoNorm(dataSet):
    minVal = np.min(dataSet,axis=0)
    maxVal = np.max(dataSet,axis=0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVal,(m,1)) # till()函数将变量复制成输入矩阵同样大小的矩阵
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVal

def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVal = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print i+1,'the classifier came back with: {0:d}, the real answer is: {1:d}'.format(classifierResult,datingLabels[i])
        if classifierResult != datingLabels[i] :
            errorCount += 1
    print 'the total error counts is {:d}'.format(errorCount)
    print 'the total error rate is: {0:f}'.format(errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    ffMiles = float(raw_input('frequen flier miles earned per year?'))
    percentTats = float(raw_input('percentage of time spent playing video games?'))
    iceCream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVal = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr - minVal)/ranges,normMat,datingLabels,4)
    print 'You will probably like this person: {:s}'.format(resultList[classifierResult - 1])


#手写识别系统
#把32x32转化为1x1024的向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits\\trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits\\trainingDigits\\%s' % fileNameStr)
    testFileList = listdir('digits\\testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits\\testDigits\\%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,4)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult,classNumStr)
        if classifierResult != classNumStr :
            errorCount += 1
    print '\nthe total number of errors is: %d' % errorCount
    print '\nthe total error rate is: %f' % (errorCount/float(mTest))
