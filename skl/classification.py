import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用于画图时显示中文

from sklearn.datasets import load_iris  # 导入数据集iris

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['花萼-length', '花萼-width', '花瓣-length', '花瓣-width', 'class']
dataset = pd.read_csv(url, names=names)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

pos = pd.DataFrame(dataset)
iris = load_iris()
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)

# 获取花卉两列数据集
L1 = pos['花萼-length'].values
L2 = pos['花萼-width'].values

import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predicted, marker='x')  # cmap=plt.cm.Paired
plt.title("DTC")
plt.show()

# 将iris_data分为70%的训练，30%的进行预测 然后进行优化 输出准确率、召回率等，优化后的完整代码如下：


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
predict_target = clf.predict(x_test)

print(sum(predict_target == y_test))  # 预测结果与真实结果比对
print(metrics.classification_report(y_test, predict_target))
print(metrics.confusion_matrix(y_test, predict_target))

L1 = [n[0] for n in x_test]
L2 = [n[1] for n in x_test]
plt.scatter(L1, L2, c=predict_target, marker='x')
plt.title('决策树分类器')
plt.show()
