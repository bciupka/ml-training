import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

flowers = datasets.load_iris()

features = flowers.data
target = flowers.target

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
tree_model = tree.DecisionTreeClassifier()

tree_model.fit(x_train, y_train)
y_pred = tree_model.predict(x_test)
tree_score = accuracy_score(y_test, y_pred)
print(tree_score)
