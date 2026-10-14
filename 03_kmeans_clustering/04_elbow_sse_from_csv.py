# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

'''
k-means最优K的选取
'''
# 專案根目錄 (此檔案的上一層)，資料路徑以專案相對路徑表示
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_Path = PROJECT_ROOT / 'data' / 'iris.csv'
df_features = pd.read_csv(data_Path)  # 读入数据

'针对特征 sepal length (cm)，利用SSE选择最优k'
SSE = []  # 存放每次结果的误差平方和
for k in range(1, 9):  # K的范围 ： 1-9
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(df_features[['sepal length (cm)']])
    SSE.append(estimator.inertia_)
X = range(1, 9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()