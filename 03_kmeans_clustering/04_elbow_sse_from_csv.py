# -*- coding: utf-8 -*-
# Elbow method on a single feature read from data/iris.csv
# 手肘法：讀取 data/iris.csv，以單一特徵選擇 k 值

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Choose the best k for k-means
# k-means 最佳 k 值的選取
# Project root (one level above this file's folder); the data path is relative to it
# 專案根目錄 (此檔案所在資料夾的上一層)，資料路徑以專案相對路徑表示
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_Path = PROJECT_ROOT / 'data' / 'iris.csv'
df_features = pd.read_csv(data_Path)  # read the data / 讀入資料

# Use SSE on the feature sepal length (cm) to choose the best k
# 針對特徵 sepal length (cm)，利用 SSE 選擇最佳 k
SSE = []  # SSE of each run / 存放每次結果的誤差平方和
for k in range(1, 9):  # k from 1 to 8 / k 的範圍：1-8
    estimator = KMeans(n_clusters=k)  # build the clusterer / 建立分群器
    estimator.fit(df_features[['sepal length (cm)']])
    SSE.append(estimator.inertia_)
X = range(1, 9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Elbow plot: SSE for k = 1-8 using only sepal length (cm) from data/iris.csv. SSE drops sharply until about k = 2-3 and then flattens; that elbow is a good choice of k.',
        '手肘圖：只用 data/iris.csv 的 sepal length (cm)，k = 1-8 時的 SSE。SSE 在 k = 2-3 左右之前快速下降、之後趨緩，手肘處即為合適的 k。'),
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