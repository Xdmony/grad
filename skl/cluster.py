import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylab import *

# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用于画图时显示中文

# from sklearn.datasets import load_iris  # 导入数据集iris

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['花萼-length', '花萼-width', '花瓣-length', '花瓣-width', 'class']
dataset = pd.read_csv(url, names=names)

pos = pd.DataFrame(dataset)

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
clf = KMeans()
clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)

pos = pd.DataFrame(dataset)
L1 = pos['花萼-length'].values
L2 = pos['花萼-width'].values

plt.scatter(L1, L2, c=predicted, marker='s', s=100, cmap=plt.cm.Paired)
plt.title("KMeans聚类分析")
plt.show()