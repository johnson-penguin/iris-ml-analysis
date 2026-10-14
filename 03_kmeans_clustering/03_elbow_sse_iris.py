# -*- coding: utf-8 -*-
# Elbow method: choose k for K-Means by SSE on all four Iris features
# 手肘法：以 SSE 選擇 K-Means 的 k 值 (使用鳶尾花全部四個特徵)

import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris

# Iris dataset
# 鳶尾花資料集
iris_dataset=load_iris()
iris = datasets.load_iris()

dx = iris_dataset['data']
dy = iris_dataset['target']

# Choose the best k for k-means
# k-means 最佳 k 值的選取
# Use SSE (sum of squared errors) to choose the best k
# 利用 SSE (誤差平方和) 選擇最佳 k
SSE = []  # SSE of each run / 存放每次結果的誤差平方和
for k in range(1, 9):  # k from 1 to 8 / k 的範圍：1-8
    estimator = KMeans(n_clusters=k)  # build the clusterer / 建立分群器
    estimator.fit(dx)
    SSE.append(estimator.inertia_)
X = range(1, 9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Elbow plot: SSE (within-cluster sum of squared errors) for k = 1-8 using all 4 features. SSE drops sharply until about k = 2-3 and then flattens; that elbow is a good choice of k.',
        '手肘圖：使用全部 4 個特徵，k = 1-8 時的 SSE (群內誤差平方和)。SSE 在 k = 2-3 左右之前快速下降、之後趨緩，手肘處即為合適的 k。'),
}
# Save every figure to the project's outputs/ folder (keeps results even without a display)
# 將所有圖存到專案的 outputs/ 資料夾 (無圖形介面時也能保留結果)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
for num in plt.get_fignums():
    out_path = OUTPUT_DIR / f'{Path(__file__).stem}_{num}.png'
    plt.figure(num).savefig(out_path, bbox_inches='tight')
    print('Saved figure / 已儲存圖檔:', out_path)
    en, zh = FIGURE_DESCRIPTIONS[num]
    print('  ' + en)
    print('  ' + zh)
plt.show()