# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:17:02 2020

@author: tgu2
"""

#!pip install surprise
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import time

"""
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/My Drive/Colab Notebooks/")
"""
time1=time.time()

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('C:\\Users\\tgu2\\Desktop\\movielens\\ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# 使用funkSVD
algo = SVD(biased=False)
# 使用biseSVD
#algo = SVD(biased=True)
# 使用SVD++
#algo = SVDpp()

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)
    
pred_tags = algo.test()

uid = str(196)
iid = str(302)
# 输出uid对iid的预测结果
pred = algo.predict(uid, iid, r_ui=4, verbose=True)
time2=time.time()
print(time2-time1)