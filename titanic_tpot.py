from sklearn.feature_extraction import DictVectorizer
from tpot import TPOTClassifier
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

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(train_features, train_labels)
print(tpot.score(test_features, test_labels))
tpot.export('tpot_titanic_pipeline.py')
