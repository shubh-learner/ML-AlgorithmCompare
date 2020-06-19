# -*- coding: utf-8 -*-

# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Importing the dataset
dataset = pd.read_csv('cleveland.csv', header = None)
dataset.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

# 1 = male, 0 = female

# Mapping and managing NaN values of the dataset
dataset['target'] = dataset.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
dataset['thal'] = dataset.thal.fillna(dataset.thal.mean())
dataset['ca'] = dataset.ca.fillna(dataset.ca.mean())

# Data-preprocessing - separating dependent and independent features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('NaiveBayes', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()
# plt.show would cause empty save image.
plt.savefig('compare-algo.png')
