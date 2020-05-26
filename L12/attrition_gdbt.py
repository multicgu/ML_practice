# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:26:54 2020

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

df_train = pd.read_csv("C:\\Users\\tgu2\\ML_practice\\L12\\train.csv")
df_test = pd.read_csv("C:\\Users\\tgu2\\ML_practice\\L12\\test.csv")

df_train["type"] = "train"
df_test["type"] = "test"

df = pd.concat([df_train,df_test],axis=0,ignore_index=True,sort=False)
df['Attrition'].fillna(0,inplace=True)

df.drop(['EmployeeCount' , 'EmployeeNumber' , 'Over18' ], axis = 1, inplace = True )

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

df_test = df_test[['user_id', 'Attrition', 'Age', 'BusinessTravel', 'DailyRate',
       'Department', 'DistanceFromHome', 'Education', 'EducationField',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

df_test = df_test.drop(['Attrition'], axis=1)
d_test = df_test.iloc[:,1:].values
uid_test = df_test['user_id'].values
#train_x = df_train.drop(['Attrition', 'user_id'], axis = 1)
#tx = train_x.values
#x_train_gbdt, x_train_lr, y_train_gbdt, y_train_lr = train_test_split(x_train, y_train, test_size=0.5)
tx = df_train.values
#tx_test = tx[0:176,:]
#tx=tx[177:,:]
#train_y = df_train['Attrition']
#ty = train_y.values

n_estimator = 50

kf = KFold(n_splits=10 , shuffle=False)
for train, test in kf.split(tx):
    train_x = tx[train][:,2:]
    train_y = tx[train][:,1]
    test_x = tx[test][:,2:]
    test_y = tx[test][:,1]

# LR
    LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    LR.fit(train_x, train_y)
    y_pred = LR.predict_proba(test_x)[:,1]
    fpr_lr, tpr_lr, _ = roc_curve(test_y, y_pred)

#gdbt+lr
    gbdt = GradientBoostingClassifier(n_estimators=n_estimator)
    gbdt.fit(train_x, train_y)
    np.set_printoptions(threshold=np.inf)  
    gbdt_enc = OneHotEncoder(categories='auto')
    gbdt_enc.fit(gbdt.apply(train_x)[:,:,0])
#    y_pred_gbdt = gbdt.predict_proba(test_x)[:, 1]
#    fpr_gbdt, tpr_gbdt, _ = roc_curve(test_y, y_pred_gbdt)
    gbdt_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    gbdt_lr.fit(gbdt_enc.transform(gbdt.apply(train_x)[:, :, 0]), train_y)
    y_pred_gbdt_lr = gbdt_lr.predict_proba(gbdt_enc.transform(gbdt.apply(test_x)[:, :, 0]))[:,1]
    fpr_gbdt_lr, tpr_gbdt_lr, _ = roc_curve(test_y, y_pred_gbdt_lr)

# gbdt
    y_pred_gbdt = gbdt.predict_proba(test_x)[:, 1]
    fpr_gbdt, tpr_gbdt, _ = roc_curve(test_y, y_pred_gbdt)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_gbdt, tpr_gbdt, label='GBDT')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBDT + LR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    

#tx_test_y = tx_test[:,1]
#tx_test_x = tx_test[:,2:]
#LR
#y_pred = LR.predict_proba(tx_test_x)[:,1]
#fpr_lr, tpr_lr, _ = roc_curve(tx_test_y, y_pred)
#gdbt+lr
y_pred_gbdt_lr = gbdt_lr.predict_proba(gbdt_enc.transform(gbdt.apply(d_test)[:, :, 0]))[:,1]
gbdt_lr_test = pd.DataFrame({'user_id':uid_test, 'Attrition':y_pred_gbdt_lr})
gbdt_lr_test.to_csv("gbdt_lr_test10.csv")
# gbdt
y_pred_gbdt = gbdt.predict_proba(d_test)[:, 1]
gbdt_test = pd.DataFrame({'user_id':uid_test, 'Attrition':y_pred_gbdt})
gbdt_test.to_csv("gt_test10.csv")

import xgboost as xgb
X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(['Attrition', 'user_id'], axis = 1), df_train['Attrition'], test_size=.2)
# 使用XGBoost
model = xgb.XGBClassifier(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42    
)
model.fit(
    X_train, y_train,
    eval_metric='auc', eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    #早停法，如果auc在10epoch没有进步就stop
    early_stopping_rounds=10 
)
model.fit(X_train, y_train)

prob = model.predict_proba(df_test.drop(['user_id'], axis = 1))
df_xgboost = pd.DataFrame({'user_id':uid_test, 'Attrition':prob[:,1]})
df_xgboost.to_csv('xgboot.csv', index=False)