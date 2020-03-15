import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris #导入数据集iris

def write_file(data):
    f = open("data.txt","w")
    for line in data:
        for x in line:
            f.write(x)
        f.write("\n")
    f.close

iris = load_iris() #载入数据集
iris.data.shape  # iris数据集150行4列的二维数组
write_file(iris)
print(iris.data)  #打印输出显示


# print(iris.target) #共150条记录，分别代表50条山鸢尾 (Iris-setosa)、变色鸢尾(Iris-versicolor)、维吉尼亚鸢尾(Iris-virginica)
#
# iris.data.shape  # iris数据集150行4列的二维数组

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pd.read_csv(url, names=names)
# dataset.hist() #数据直方图histograms
#
# print(dataset.describe())
#
# dataset.plot(x='sepal-length', y='sepal-width', kind='scatter') #散点图，x轴表示sepal-length花萼长度，y轴表示sepal-width花萼宽度
#
# dataset.plot(kind='kde') #KDE图，KDE图也被称作密度图(Kernel Density Estimate,核密度估计)
#
# #kind='box'绘制箱图,包含子图且子图的行列布局layout为2*2，子图共用x轴、y轴刻度，标签为False
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)