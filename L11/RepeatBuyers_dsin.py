# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:42:21 2020

@author: tgu2
"""
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.inputs import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names
import pandas as pd

df = pd.read_csv("C:\\Users\\tgu2\\.spyder-py3\\Criteo.csv")  
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1,14)]

df[sparse_features] = df[sparse_features].fillna('-1',)
df[dense_features] = df[dense_features].fillna('0',)
target = ['label']

for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])
 
fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique()) for feat in sparse_features] + [DenseFeat(feat, 1,) for feat in dense_features]

aa = VarLenSparseFeat(fixlen_feature_columns[0],100)