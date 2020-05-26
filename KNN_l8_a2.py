# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:32:23 2020
"""

from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
from surprise.model_selection import KFold
from surprise import accuracy

# 数据读取
#movies = pd.read_csv("C:\\Users\\tgu2\\RS6-master\\L5\\MovieLens\\movies.csv", encoding = 'gbk' )
#rating = pd.read_csv("C:\\Users\\tgu2\\RS6-master\\L5\\MovieLens\\ratings.csv")
#df = rating.merge(movies, how='inner', on='movieId')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('C:\\Users\\tgu2\\RS6-master\\L5\\MovieLens\\ratings.csv', reader=reader)

trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})
kf = KFold( n_splits = 3 )
for a, b in kf.split(data):
    algo.fit(a)
    predictions = algo.test(b)
    accuracy.rmse(predictions, verbose = True)
    accuracy.mae(predictions, verbose = True)
#    perf = evaluate(algo, a, measures=['RMSE', 'MAE'])
#    print_perf(perf)
#本来想用evaluate,但好像一直失败。。。
    

algo.fit(trainset)

uid = str(196)
iid = str(302)

pred = algo.predict(uid, iid)
print(pred)
