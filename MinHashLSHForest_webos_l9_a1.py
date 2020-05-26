# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:47:22 2020
"""

from datasketch import MinHash, MinHashLSH, MinHashLSHForest
import pandas as pd
import jieba as jb

f = open("C:\\Users\\tgu2\\RS6-master\\L9\\weibos.txt", encoding='utf-8')
fl = f.readlines()
f.close()

# For except the bd
bd = [',','。','，','\n','\u200b','#','、','：','—','_','-','！','…','”','？','“','”',' ','《','》']
for l in range(len(fl)):
    for b in bd:
        fl[l] = fl[l].replace(b, '')

# cut every sentense to word to the list d by jieba
data = []
for i in fl:
    data.append(jb.lcut(i,cut_all=False))

# create the minihash for every sentense.
m = []
for i in range(len(fl)):
    m.append(MinHash())

for i,d in enumerate(data):
    for di in d:
        m[i].update(di.encode('utf-8'))

# 创建LSH Forest
forest = MinHashLSHForest()
for i in range(len(m)-1):
    forest.add(i,m[i])

# 在检索前，需要使用index
forest.index()
# 判断forest是否存在1, 2
print(1 in forest)
print(4 in forest)
# 查询的句子
print("查询的句子: ", fl[-1])
# 查询forest中与m1相似的Top-K个邻居
result = forest.query(m[-1], 3)
print("Top 3 相似:\n", result)

for i in result:
    print(fl[i])
    