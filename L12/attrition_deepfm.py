# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:00:17 2020

@author: tgu2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat,get_feature_names

df_train = pd.read_csv("C:\\Users\\tgu2\\ML_practice\\L12\\train.csv")
df_test = pd.read_csv("C:\\Users\\tgu2\\ML_practice\\L12\\test.csv")
lbe_Attrition = LabelEncoder()
df_train['Attrition']  = lbe_Attrition.fit_transform(df_train['Attrition'])
train_tar = df_train['Attrition']
target = ['Attrition']
uid_test = df_test['user_id'].values

df_train["type"] = "train"
df_test["type"] = "test"

df = pd.concat([df_train,df_test],axis=0,ignore_index=True,sort=False)
df['Attrition'].fillna(2,inplace=True)

df.drop(['EmployeeCount' , 'EmployeeNumber' , 'Over18' ], axis = 1, inplace = True )
'''
lbe_BusTra = LabelEncoder()
df['BusinessTravel'] = lbe_BusTra.fit_transform(df['BusinessTravel'])

lbe_Dep = LabelEncoder()
df['Department'] = lbe_Dep.fit_transform(df['Department'])

lbe_EduF = LabelEncoder()
df['EducationField'] = lbe_EduF.fit_transform(df['EducationField'])

lbe_Gen = LabelEncoder()
df['Gender'] = lbe_Gen.fit_transform(df['Gender'])

lbe_JobRole = LabelEncoder()
df['JobRole']  = lbe_JobRole.fit_transform(df['JobRole'])

lbe_Marital = LabelEncoder()
df['MaritalStatus'] = lbe_Marital.fit_transform(df['MaritalStatus'])

lbe_OverTime = LabelEncoder()
df['OverTime'] = lbe_OverTime.fit_transform(df['OverTime'])

df_train = df[df['type'] == 'train'].drop(['type'], axis=1)
df_test = df[df['type'] == 'test' ].drop(['type'], axis=1)

lbe_Attrition = LabelEncoder()
df_train['Attrition']  = lbe_Attrition.fit_transform(df_train['Attrition'])
'''

df_train = df_train[['user_id', 'Attrition', 'Age', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

df_test = df_test[['user_id', 'Age', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

sparse_features = ['user_id', 'Age', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

for feature in sparse_features:
    lbe = LabelEncoder()
    df[feature] = lbe.fit_transform(df[feature])
# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, df[feature].nunique()) for feature in sparse_features]
print(fixlen_feature_columns)
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

df_train = df[ df['type'] =='train'].drop(['type'], axis=1)
df_test = df[ df['type'] =='test'].drop(['type'], axis=1)
# 将数据集切分成训练集和测试集
train, test = train_test_split(df_train, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

# 使用DeepFM进行训练
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=True, validation_split=0.2, )
# 使用DeepFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
testrr = {name:df_test[name].values for name in feature_names}
pred = model.predict(testrr, batch_size=256)
deepfmrrr = pd.DataFrame({'user_id':uid_test, 'Attrition':pred[:,0]})
deepfmrrr.to_csv('df.csv', index = False)

