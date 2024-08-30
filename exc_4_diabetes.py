import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

features = pd.read_csv("./resources/diabetes.csv")
target = features.loc[:, "Outcome"]
features.drop("Outcome", inplace=True, axis=1)

print(features.head())
print(target.head())

scores = []
count = 300
for i in range(count):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=i, stratify=target
    )
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = accuracy_score(y_pred=prediction, y_true=y_test)
    scores.append(score)
    pickle.dump(model, open(f"./exc_4_models/{i}", "wb"))

print(np.mean(scores), np.std(scores))

scores = []
for i in os.listdir("./exc_4_models"):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=int(i), stratify=target
    )
    model = pickle.load(open(f"./exc_4_models/{i}", "rb"))
    prediction = model.predict(x_test)
    score = accuracy_score(y_pred=prediction, y_true=y_test)
    scores.append(score)

print(np.mean(scores), np.std(scores))

averages = []
for i in range(count):
    averages.append(np.mean(scores[: i + 1]))

plt.plot(range(1, 301), averages)
plt.show()
