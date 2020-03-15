# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyecharts.charts import Scatter

# from pylab import *
#
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用于画图时显示中文

from sklearn.datasets import load_iris  # 导入数据集iris

# iris = load_iris()  # 载入数据集
# print(iris.data)  # 打印输出数据集
#
# # 共150条记录，分别代表50条山鸢尾 (Iris-setosa)、变色鸢尾(Iris-versicolor)、维吉尼亚鸢尾(Iris-virginica)
# print(iris.target)
#
# iris.data.shape  # iris数据集150行4列的二维数组

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['花萼-length', '花萼-width', '花瓣-length', '花瓣-width', 'class']
dataset = pd.read_csv(url,names= names)

# *****************************线性回归*************************************#
print(dataset)
pos = pd.DataFrame(dataset)
# 获取花瓣的长和宽，转换Series为ndarray
x = pos['花瓣-length'].values
y = pos['花瓣-width'].values
x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x, y)
pre = clf.predict(x)

plt.scatter(x, y, s=100)
plt.plot(x, pre, 'r-', linewidth=4)
for idx, m in enumerate(x):
    plt.plot([m, m], [y[idx], pre[idx]], 'g-')
plt.title("线性回归")
plt.show()
#
# Scatter.add_xaxis(x,y)
# Scatter.add_yaxis(x,pre)
# Scatter.render("Iris")
