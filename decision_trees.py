from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

tree =  DecisionTreeClassifier(max_depth = 2)
tree.fit(x, y)


"""Supposing we wanna predict a new flower whose patals are 5cm long and 1.5cm wide"""
prediction = tree.predict([[5, 1.5]])
print(prediction)


"""Supposing we wanna see the see the probabilities for our predictions"""
probabilty_prediction = tree.predict_proba([[5,1.5]])
print(probabilty_prediction)


"""The decision tree itself"""
export_graphviz(
    tree,
    out_file = "iris_decision_tree.png",
    feature_names = iris.feature_names[2:],
    class_names = iris.target_names,
    rounded = True,
    filled = True
    )
