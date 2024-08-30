import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

X = x_train[:2000]
y = y_train[:2000]

model = KNeighborsClassifier()

print(cross_val_score(model, X, y, cv=3))
