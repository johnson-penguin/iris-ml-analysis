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

# Figure 2: k = 3 only, sepal vs. petal; color = true species, black X = sample put in a cluster of another species
# 圖 2：只看 k = 3，花萼 vs 花瓣；顏色 = 實際品種，黑色 X = 被分到其他品種為主的群的樣本
species_colors = ['red', 'green', 'blue']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
print('k=3 misclassified / 分錯筆數:')
for ax, (name, cols, xlabel, ylabel) in zip(axes, FEATURE_SETS):
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X[:, cols])
    # Map each cluster to its majority species, then compare with the true species
    # 把每一群對應到群內最多的品種，再與實際品種比較
    cluster_to_label = np.array([np.bincount(y[kmeans.labels_ == c]).argmax() for c in range(3)])
    wrong = cluster_to_label[kmeans.labels_] != y
    print(f'  {name}: {wrong.sum()} / {len(y)}  (purity {1 - wrong.mean():.1%})')
    for i, color in enumerate(species_colors):
        ax.scatter(X[y == i, cols[0]], X[y == i, cols[1]], color=color, alpha=0.6, label=iris.target_names[i])
    ax.scatter(X[wrong, cols[0]], X[wrong, cols[1]], marker='x', color='black', s=80, linewidth=2,
               label=f'misclassified ({wrong.sum()})')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', color='gold',
               edgecolor='black', s=300, label='cluster center')
    ax.set_title(f'K-Means k=3 on {name.lower()} features: {wrong.sum()} misclassified, purity={1 - wrong.mean():.1%}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8)
plt.tight_layout()
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: (f'Top row: original species; other rows: K-Means with k = {K_LIST}. Left column clusters on sepal length/width only, right column on petal length/width only. Petal clusters have clean borders, while sepal clusters cut through the overlapping versicolor/virginica region.',
        f'第一列：原始品種；其餘各列：k = {K_LIST} 的 K-Means 結果。左欄只用花萼長度/寬度分群，右欄只用花瓣長度/寬度分群。花瓣分群的邊界清楚，花萼分群則在 versicolor/virginica 的重疊區被任意切開。'),
    2: ('K-Means with k = 3, sepal features only (left) vs. petal features only (right). Color = true species, black X = sample placed in a cluster dominated by another species, star = cluster center. Sepal clustering misclassifies 27 samples, petal clustering only 6, all on the versicolor/virginica border.',
        'k = 3 的 K-Means，只用花萼特徵 (左) vs 只用花瓣特徵 (右)。顏色 = 實際品種，黑色 X = 被分到其他品種為主的群的樣本，星號 = 群中心。花萼分群分錯 27 筆，花瓣分群只分錯 6 筆，且都在 versicolor/virginica 的交界。'),
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
