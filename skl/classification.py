from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)

# 获取花卉两列数据集
L1 = pos['sepal-length'].values
L2 = pos['sepal-width'].values

import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predicted, marker='x')  # cmap=plt.cm.Paired
plt.title("DTC")
plt.show()   