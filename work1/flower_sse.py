# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris

#鳶尾花資料集
iris_dataset=load_iris()
iris = datasets.load_iris()

dx = iris_dataset['data']
dy = iris_dataset['target']

'''
k-means最优K的选取
'''
'针对对特征Parameter1，利用SSE选择最优k'
SSE = []  # 存放每次结果的误差平方和
for k in range(1, 9):  # K的范围 ： 1-9
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(dx)
    SSE.append(estimator.inertia_)
X = range(1, 9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()