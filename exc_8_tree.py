import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

X = x_train[:2000]
y = y_train[:2000]

model = DecisionTreeClassifier()

params = {"max_depth": [4, 6, 8], "min_samples_leaf": [1, 3, 5]}
grid = GridSearchCV(model, params)

grid.fit(X, y)

print(grid.best_params_, "/n", grid.best_score_)
