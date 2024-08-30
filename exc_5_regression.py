from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


features = pd.read_csv("./resources/diabetes.csv")
target = features.loc[:, "Age"]
features.drop("Age", axis=1, inplace=True)

count = 30
scores = []
for i in range(count):
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=i
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    score = mean_absolute_error(y_pred=prediction, y_true=y_test)
    scores.append(score)

print(np.mean(scores), np.std(scores))
