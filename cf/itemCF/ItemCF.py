#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cf.Data as Data
import json

class ItemCF:

    """
    基于物品的协同过滤算法.
    """

    def __init__(self, fileName, N = 10):
        self.data = Data.loadFormatData(fileName)
        self.itemMatrix = self.item_Matrix(N)

    def cos_similiarity(self, iUsers, jUsers):

        """
        计算两个物品对之间的相似度.
        采用余弦地计算方式.
        这里针对活跃用户进行了一定地惩罚.
        1/log(1.0 + N(u)) N(u)代表用户已经购买的物品总数.
        用户越活跃，则相似度地贡献越小.
        """
        iSquare = np.square(iUsers).sum()
        jSquare = np.square(jUsers).sum()
        return iUsers.mul(jUsers,fill_value = 0).mul(1/np.log(self.data['itemId'].groupby(self.data['userId']).count() + 1.0),fill_value = 0).sum()/np.sqrt(iSquare * jSquare)


    def item_Matrix(self, N = 10):
        """
        计算物品的相似性矩阵.
        """

        data = self.data.set_index('userId') #注意这里需要以userId作为其row_index.
        groupbyItemId = dict(list(data.groupby('itemId')))

        itemsMatrix = dict()

        for i in groupbyItemId.keys():
            itemsMatrix.setdefault(i,dict())
            for j in groupbyItemId.keys():
                if i == j:
                    continue

                sim = self.cos_similiarity(groupbyItemId[i]['rates'],groupbyItemId[j]['rates'])
                if sim == 0:
                    continue
                itemsMatrix[i][j] = sim

        return itemsMatrix


    def recommendAll(self, N = 10):

        groupByUserId = dict(list(self.data.groupby('userId')))

        recommendResultMatrix = dict()
        for u in groupByUserId.keys():
            recommendResultMatrix.setdefault(u,dict())
            for index, row in groupByUserId[u].iterrows():
                if row.itemId not in self.itemMatrix:
                    continue

                #查找与每个物品最相近的N个物品.
                for j, itemSims in sorted(self.itemMatrix[row.itemId].items(),key= lambda x:x[1],reverse=True)[:N]:
                    if j in groupByUserId[u]['itemId']:
                        continue

                    recommendResultMatrix[u].setdefault(j,[0.0,0.0])
                    recommendResultMatrix[u][j] += itemSims
                    recommendResultMatrix[u][j] += itemSims * row.rates


        for u,recommends in recommendResultMatrix.iteritems():
            for item,simsAndScores in recommends.iteritems():
                if simsAndScores[0] == 0:
                    recommendResultMatrix[u][item] = 0.0
                else:
                    recommendResultMatrix[u][item] = simsAndScores[1]/simsAndScores[0]

        return recommendResultMatrix

    def writeToFile(self,fileName,N = 10):

        """
        将推荐结果写入到文件中.
        """
        recommendResult = self.recommendAll(N)
        fw = open(fileName,'w')
        json.dump(recommendResult,fw)
        fw.close()
        return fileName