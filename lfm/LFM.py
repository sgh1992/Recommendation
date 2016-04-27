#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import cf.Data as Data
import numpy as np
import json
import math

class LFM:
    """
    这里采用隐语义模型来进行推荐.
    preference(u,i) = pu * qk
    pu为一个 |U| * K 的矩阵.表示每个用户对各个隐含类别的兴趣度.
    qk为一个K *|Item|的矩阵，表示每个物品在各个类别的权重大小.
    那么，这里本质上就是一个有学习过程的推荐算法.
    就是通过训练数据集来最小化sum(rui - tui)^2 评分的最小误差值来获得其参数.
    从而，根据其训练出来的参数来对用户进行预测得分.
    这个算法最终没有收敛.
    P,Q矩阵会变得无穷大.
    """

    def __init__(self, fileName, K=3, learningRate=0.9, alpha=0.000001, iterNums=1000):
        """
        alpha:正则化系数.
        learningRate 梯度下降时的学习率.
        K 隐含类别的个数.
        """
        self.data = Data.loadFormatData(fileName)
        self.K = K
        self.learningRate = learningRate
        self.alpha = alpha
        # 以下分别为pu或者qk矩阵.注意，这里直接用ndarray数据结构来表征这些数据.
        # 因此用户与物品的id必须是整数.这里原本需要一个映射表.userName->userId,itemName->itemId,只不过数据已经映射过了.
        self.puMatrix = np.ones((len(self.data['userId'].unique()) + 1, K))  # 注意这里的下标是从1开始的，因此数组的长度为N + 1
        self.qkMatrix = np.ones((K, len(self.data['itemId'].unique()) + 1))
        self.gradientDescent(iterNums)

    def gradientDescent(self, iterNums=1000):
        """
        使用梯度下降的方法来计算优化出pu矩阵与qk矩阵的参数值.
        """
        for iter in range(iterNums):
            squareError = 0.0
            for index, row in self.data.iterrows():

                predictResult = self.predict(row.userId, row.itemId)
                if predictResult > 37:
                    print 'Debug'
                #if math.fabs(predictResult) > 100:
                #    if predictResult > 0:
                #        predictResult = 100
                #    else:
                #        predictResult = -100
                ei = row.rates - predictResult
                if row.itemId == 382:
                    print 'Debug'
                #print 'real:%f\t predict: %f\t index:%d' % (row.rates, predictResult, index)
                #print ei
                squareError += ei * ei
                for k in range(self.K):
                    self.puMatrix[row.userId, k] += self.learningRate * (
                        self.qkMatrix[k, row.itemId] * ei - self.alpha * self.puMatrix[row.userId, k])
                    print 'pu', self.puMatrix[row.userId,k]
                    #if math.fabs(self.puMatrix[row.userId, k]) >= 10000:
                    #    if self.puMatrix[row.userId, k] > 0:
                    #        self.puMatrix[row.userId, k] = 10000
                    #    else:
                    #        self.puMatrix[row.userId, k] = -10000

                    self.qkMatrix[k, row.itemId] += self.learningRate * (
                        self.puMatrix[row.userId, k] * ei - self.alpha * self.qkMatrix[k, row.itemId])
                    print 'qk', self.qkMatrix[k, row.itemId]

                    #self.puMatrix[row.userId, k] = puk
                    #self.qkMatrix[k, row.itemId] = qki
                    #if math.fabs(self.qkMatrix[k, row.itemId]) >= 10000:
                    #    if self.qkMatrix[k,row.itemId] > 0:
                    #        self.qkMatrix[k, row.itemId] = 10000
                    #    else:
                    #       self.qkMatrix[k, row.itemId] = -10000
                if row.userId == 90:
                    print 'pu of userId:90',self.puMatrix[row.userId]

                if row.itemId == 382:
                    print 'qk of itemId:382', self.qkMatrix[:, row.itemId]

            if iter % 2 == 0:
                print 'current iter:%d,sumError:%s' % (iter, squareError)
            if squareError <= 1e-6:
                break

    def predict(self, userId, itemId):
        """
        返回预测的值.
        根据userId,itemId的结果
        以及self.pu,self.qk
        """
        result = self.puMatrix[userId].dot(self.qkMatrix[:, itemId])
        #print "pu", self.puMatrix[userId]
        #print "qk", self.qkMatrix[:, itemId]
        if result == 4427836186.078808:
            sys.exit()
        if math.isnan(result) or math.isinf(result):
            print self.puMatrix
            sys.exit()
        return result

    def recommendForUser(self, userId, N=16):

        """
        对某一个用户进行推荐，并返回TopN的推荐结果
        """
        data = dict(list(self.data.groupby('userId')))
        recommendResult = dict()
        for item in self.data['itemId']:
            if item in data[userId]['itemId']:
                continue
            recommendResult[item] = self.predict(userId, item)
        return sorted(recommendResult.iteritems(), key=lambda x: x[1], reverse=True)[:N]

    def recommendAll(self, N=16):
        """
        针对所有用户进行推荐.
        """
        data = dict(list(self.data.groupby('userId')))
        recommendResult = dict()

        for userId in recommendResult.keys():

            recommendResult.setdefault(userId, dict())
            for item in self.data['itemId']:
                if item in data[userId]['itemId']:
                    continue
                recommendResult[userId][item] = self.puMatrix[userId].dot(self.qkMatrix[:, item])

            recommendResult[userId] = sorted(recommendResult[userId].iteritems(), key=lambda x: x[1], reverse=True)[:N]

        return recommendResult

    def writeToFile(self, fileName, N=16):
        """
        将推荐结果写入到文件中.
        """
        fw = open(fileName, 'w')
        json.dump(self.recommendAll(N), fw)
        fw.close()
        return fileName



if __name__ == 'main':
    print 'Hello World'
    LFM('/home/sghipr/Downloads/ml-100k/u.data')
