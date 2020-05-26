# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:11:29 2020

@author: tgu2
"""

import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise import BaselineOnly, KNNBasic, NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold

# this function only for netflix train set 
def pd_trainset_clean(path,col):
    df = pd.read_csv(path,header=None ,names = col)
    col.insert(1,'mid')
    df = df.reindex(columns = col)
    #df['empty']=[]
    df.mid = df.loc[:,['uid','mid']][df.uid.str.contains(':')].fillna(method='ffill',axis=1).mid
    df.mid.fillna(method='ffill',inplace=True)
    df.dropna(how='any',inplace = True)
    df.mid = df.mid.str.split(':').str[0]
    return df

# 这里是将4个数据集变成DataFrame format,并调整成4列uid，mid，rating，year
#df_c1 = pd_trainset_clean("C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_1.txt",['uid','rating','year'])
#df_c2 = pd_trainset_clean("C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_2.txt",['uid','rating','year'])
#df_c3 = pd_trainset_clean("C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_3.txt",['uid','rating','year'])
#df_c4 = pd_trainset_clean("C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_4.txt",['uid','rating','year'])
'''
#For calculate the train set RMSE, just calculate part one.
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df_c1[['uid','mid','rating']].head(int(len(df_c1)/100)), reader)
trainset = data.build_full_trainset()

# ALS优化
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
# SGD优化
#bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)
''' 

# probe set predict
# import probe，并调整为DataFrame的格式，列为uid，mid.
df_probe = pd_trainset_clean("C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\probe.txt",['uid'])
#for del : rows
pua = list(df_probe.uid)
pu=list(df_probe.uid[df_probe.uid.str.contains(':')])
rest = list(set(pua) ^ set(pu))
df_probe = df_probe[df_probe.uid.isin(rest)]
df_probe = df_probe.reset_index(drop=True )
probe_pred = []
# for rmse, [(uid,mid,rating)]


probe = []

'''
# Loop all train set to get rating value for probe
raw_t = ["C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_1.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_2.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_3.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_4.txt",]
for rt in raw_t:
    df = pd_trainset_clean(rt,['uid','rating','year'])
    for p in df_probe.itertuples():
        for t in df_c1.itertuples():
            if ( p[2] == t[2] ) & ( p[1] == t[1] ):
                probe.append(t[3])
        
df_probe['rating'] = probe
'''

'''
# Only use one train set to get rating value for probe
raw_t = ["C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_1.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_2.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_3.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_4.txt",]
df = pd_trainset_clean(raw_t[0],['uid','rating','year'])
for p in df_probe.itertuples():
    for t in df_c1.itertuples():
        if ( p[2] == t[2] ) & ( p[1] == t[1] ):
            probe.append(t[3])
        
df_probe['rating'] = probe
'''

# Only use part of one train set (len of probe) to get rating value for probe
raw_t = ["C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_1.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_2.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_3.txt",
      "C:\\Users\\tgu2\\Desktop\\netflix-prize-data\\combined_data_4.txt",]
# 只导入第一个train set
df = pd_trainset_clean(raw_t[0],['uid','rating','year'])
#由于数据集太大，将第一个train set的大小调整至probe大小的1/10，取最前面。
df = df.loc[0:len(df_probe)/10,:]
# 数据集太大，将probe的大小缩小100倍，取最前面。
df_probe = df_probe.loc[0:len(df_probe)/100,:]
#将找到调整后的train set中和probe对应的rating。
for p in df_probe.itertuples():
    for t in df.itertuples():
        if ( p[2] == t[2] ) & ( p[1] == t[1] ):
            probe.append(t[3])
        
df_p = df_probe.loc[0:len(probe)-1,:]
df_p['rating'] = probe


#For calulate the probe RMSE

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(df_p[['uid','mid','rating']], reader)
trainset = data.build_full_trainset()

# ALS优化
bsl_options = {'method': 'als','n_epochs': 5,'reg_u': 12,'reg_i': 5}
# SGD优化
#bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)



#predictions_probe = algo.predict(probe)

