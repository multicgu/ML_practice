# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:10:43 2020

@author: tgu2
"""


import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from deepctr.models import xDeepFM
from deepctr.inputs import SparseFeat,get_feature_names
chunk = pd.read_csv("C:\\Users\\tgu2\\ML_practice\\L15\\chunk.csv", dtype={'id':str} )


sparse_features = ['C1', 'banner_pos', 'site_domain', 'site_id','site_category','app_id','app_category', 'device_type', 'device_conn_type','C14', 'C15','C16']
target = ['click']


for feature in sparse_features:
    lbe = LabelEncoder()
    chunk[feature] = lbe.fit_transform(chunk[feature])
fixlen_feature_columns = [SparseFeat(feature, chunk[feature].nunique()) for feature in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(fixlen_feature_columns)
print(feature_names)

train, test = train_test_split(chunk, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}
    
model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
    
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print("test RMSE", rmse)


# 输出LogLoss
from sklearn.metrics import log_loss
score = log_loss(test[target].values, pred_ans)
print("LogLoss", score)
