from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

features = data.data
target = data.target

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=123
)

model = KNeighborsClassifier()
for i in range(features.shape[1]):
    model.fit(x_train[:, : i + 1], y_train)
    y_pred = model.predict(x_test[:, : i + 1])
    score = accuracy_score(y_test, y_pred)
    print(f"{i+1} columns: {round(score, 3)}")
