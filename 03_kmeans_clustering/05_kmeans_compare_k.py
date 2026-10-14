# Compare K-Means results for several k: cluster on sepal features only vs. petal features only
# 比較不同 k 的 K-Means 結果：只用花萼特徵分群 vs 只用花瓣特徵分群

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Number of clusters to compare
# 要比較的分群數
K_LIST = [2, 3, 4, 6, 8]

# Feature pairs: columns 0-1 are sepal length/width, columns 2-3 are petal length/width
# 特徵組合：第 0-1 欄為花萼長度/寬度，第 2-3 欄為花瓣長度/寬度
FEATURE_SETS = [
    ('Sepal', [0, 1], 'sepal length (cm)', 'sepal width (cm)'),
    ('Petal', [2, 3], 'petal length (cm)', 'petal width (cm)'),
]


# Purity: map each cluster to its majority species and count how many samples match
# 純度：把每一群對應到群內最多的品種，計算有多少樣本符合
def purity(labels):
    return sum(np.bincount(y[labels == c]).max() for c in np.unique(labels)) / len(y)


plt.rcParams['font.size'] = 9
fig, axes = plt.subplots(len(K_LIST) + 1, 2, figsize=(10, 4 * (len(K_LIST) + 1)))

# First row: original species labels
# 第一列：原始品種標籤
for col, (name, cols, xlabel, ylabel) in enumerate(FEATURE_SETS):
    ax = axes[0, col]
    ax.scatter(X[:, cols[0]], X[:, cols[1]], c=y, cmap=plt.cm.Set1)
    ax.set_title(f'Original species ({name})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Other rows: K-Means on only the two features shown, fixed random_state for reproducible results
# 其餘各列：只用圖上的兩個特徵做 K-Means，固定 random_state 讓結果可重現
print('k | sepal purity | petal purity / 花萼純度 | 花瓣純度')
for row, k in enumerate(K_LIST, start=1):
    scores = []
    for col, (name, cols, xlabel, ylabel) in enumerate(FEATURE_SETS):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X[:, cols])
        scores.append(purity(labels))
        ax = axes[row, col]
        ax.scatter(X[:, cols[0]], X[:, cols[1]], c=labels, cmap=plt.cm.Set1)
        ax.set_title(f'K-Means k={k} ({name}), purity={scores[-1]:.1%}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    print(f'{k} | {scores[0]:.1%} | {scores[1]:.1%}')

plt.tight_layout()
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: (f'Top row: original species; other rows: K-Means with k = {K_LIST}. Left column clusters on sepal length/width only, right column on petal length/width only. Petal clusters have clean borders, while sepal clusters cut through the overlapping versicolor/virginica region.',
        f'第一列：原始品種；其餘各列：k = {K_LIST} 的 K-Means 結果。左欄只用花萼長度/寬度分群，右欄只用花瓣長度/寬度分群。花瓣分群的邊界清楚，花萼分群則在 versicolor/virginica 的重疊區被任意切開。'),
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
