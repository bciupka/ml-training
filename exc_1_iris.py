from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

features = data.data
target = data.target

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=123
)

tree_model = DecisionTreeClassifier()
mlp_model = MLPClassifier(max_iter=2000)
forest_model = RandomForestClassifier(n_estimators=100)
neighbors_model = KNeighborsClassifier()

models = [tree_model, mlp_model, forest_model, neighbors_model]
names = ["Decision tree", "MLP model", "Random forest", "K-Neighbors"]

for model, name in zip(models, names):
    results = []
    for _ in range(10):
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        score = accuracy_score(y_test, prediction)
        results.append(round(score, 3))
    scores = str(results).replace(",", "\t")
    print(f"{name}:\t {scores[1:-1]}")
