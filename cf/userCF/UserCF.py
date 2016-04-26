#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cf.Data as Data
import math
import numpy as np
import json

class UserCF:
    """
    基于用户的协同过滤算法.
    注意，整个算法整体的设计结构.
    从面向对象的角度来整体设计
    """

    def __init__(self, fileName):
        self.data = Data.loadFormatData(fileName)
        self.users_matrix = self.matrix_UserSimilarity(self.data)

    def cos_similiaritySeries(self, uItems, vItems):

        """
       利用pandas的数据结构来进行计算两个用户的相似性.
       相对于热门物品，冷门物品对用户的相似度的重要性要更高.
       即通过1/(log(1.0 + N(i))来惩罚热门物品
       N(i)表示购买物品i的总人数，购买人数越多，相似度越小.
       :param uItems:
       :param vItems:
       :return:
       """
        squareU = np.square(uItems).sum()
        squareV = np.square(vItems).sum()
        return uItems.mul(vItems,fill_value = 0).mul(1/np.log(self.data['userId'].groupby(self.data['itemId']).count() + 1.0), fill_value = 0).sum()/np.sqrt(squareU * squareV)

    def matrix_UserSimilarity(self, data):

        """
      计算用户的相似性矩阵.
      """
        data = data.set_index('itemId')
        groupByUserId = dict(list(data.groupby('userId')))
        users_matrix = dict()

        for u in groupByUserId.keys():
            users_matrix.setdefault(u, dict())
            for v in groupByUserId.keys():
                if u == v:
                    continue
                sim = self.cos_similiaritySeries(groupByUserId[u]['rates'], groupByUserId[v]['rates'])
                # 如果两者之间的相似度为0，则不需要记录.
                if sim == 0.0:
                    continue
                users_matrix[u][v] = sim
        return users_matrix

    def recommendAll(self, N=10):
        """
        对所有的用户进行TopN推荐.
        注意,这里是离线地进行推荐.
        """
        recommendMatrix = dict()
        groupbyUserId = dict(list(self.data.groupby('userId')))
        for u in groupbyUserId.keys():
            similiarys = sorted(self.users_matrix[u].items(), key=lambda x: x[1], reverse=True)
            recommendMatrix.setdefault(u, dict())
            for v, sim in similiarys[:N]:
                if sim == 0:
                    continue

                for index, row in groupbyUserId[v].iterrows():
                    if row.itemId not in groupbyUserId[u]['itemId']:
                        recommendMatrix[u].setdefault(row.itemId, [0.0, 0.0])  # 总的相似度之和,与总的相似度与评分之和；用于计算加权推荐评分.
                        recommendMatrix[u][row.itemId][0] += sim
                        recommendMatrix[u][row.itemId][1] += sim * row.rates
        #计算最终的加权评分.
        for u, recommends in recommendMatrix.iteritems():
            for item, simsAndScores in recommends.iteritems():
                if simsAndScores[0] != 0:
                    recommendMatrix[u][item] = simsAndScores[1] / simsAndScores[0]
                else:
                    recommendMatrix[u][item] = 0
            #python中的字典对象并不保证排序.如果它返回的是一个字典对象的话，排序没有任何用处.
            #recommendMatrix[u] = dict(sorted(recommendMatrix[u].iteritems(), key=lambda x: x[1], reverse=True))
        return recommendMatrix

    def recommendForUser(self, u, N=10):
        """
        针对某一个用户进行推荐.
        """
        groupByUserId = dict(list(self.data.groupby('userId')))
        if u not in groupByUserId:
            print '%s not in userLists' % (u)
            return

        recommendResult = dict()
        for v, sim in self.users_matrix[u].items():
            if sim == 0.0:
                break

            for index, row in groupByUserId[v].iterrows():
                if row.itemId in groupByUserId[u]['itemId']:
                    continue
                recommendResult.setdefault(row.itemId, [0.0, 0.0])
                recommendResult[row.itemId][0] += sim
                recommendResult[row.itemId][1] += sim * row.rates

        for item, simsAndScores in recommendResult.iteritems():
            if simsAndScores[0] != 0:
                recommendResult[item] = simsAndScores[1] / simsAndScores[0]
            else:
                recommendResult[item] = 0.0

        return dict(sorted(recommendResult.iteritems(),key= lambda x:x[1],reverse=True))


    def writeRecommendResult(self, fileName, N = 10):
        """
        将结果写入到文件中.以字典的形式写入.
        其实也是一种序列化的过程.
        """
        f = open(fileName,'w')
        recommendResultMatrix = self.recommendAll(N)
        json.dump(recommendResultMatrix,f)
        f.close()
        return fileName