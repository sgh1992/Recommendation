#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

files = '/home/sghipr/Downloads/ml-100k/u.data'

def loadData(fileName = files):
    """
    数据格式: userId   itemId  rates   timestamp
    """
    lines = [row.strip().split('\t') for row in open(fileName)]
    data = []
    for line in lines:
        values = []
        for value in line[:-1]:
            values.append(value)
        data.append(values)
    return data


def loadFormatData(fileName = files):

    """
    使用pandas来读取数据.
    """
    return pd.read_csv(fileName,sep="\t",names=['userId','itemId','rates','timestamp'],usecols=['userId','itemId','rates'])
