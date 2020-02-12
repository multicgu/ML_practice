from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

train_df= pd.read_csv("C://Users/tgu2//.spyder-py3//Titanic_Data-master\\train.csv")
test_df = pd.read_csv("C://Users/tgu2//.spyder-py3//Titanic_Data-master\\test.csv")

train_df.Age.fillna(train_df.Age.mean(), inplace=True)
test_df.Age.fillna(train_df.Age.mean(), inplace=True)

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)

train_df.Embarked.fillna('S', inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_df[features]
#train_labels = train_df['Survived']
train_labels = train_df['Survived'].as_matrix()

test_features = test_df[features]
test_labels= np.array([0])

dvec = DictVectorizer(sparse = False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)
test_features = dvec.fit_transform(test_features.to_dict(orient='record'))
print(dvec.feature_names_)

lr = LogisticRegression()
lr.fit(train_features, train_labels)
predict=lr.predict(test_features)
x_predict = lr.predict(train_features)

def acc(y_target, y_predict):
    return (y_target==y_predict).mean()
a = acc(train_labels,x_predict)

print("LR 训练集的准确率为： %0.4lf" % a)
#print("LR 准确率为： %0.4lf" % accuracy_score(train_features, train_labels))
print("LR 准确率为： %0.4lf" % accuracy_score(train_labels, x_predict))