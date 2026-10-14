# Decision boundaries of KNN (k=1, k=17), LDA and K-Means on the two petal features, with Chinese labels
# 以兩個花瓣特徵畫出 KNN (k=1、k=17)、LDA 與 K-Means 的決策邊界，圖與輸出皆使用中文

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Chinese font: use the first one available (Linux / Windows / macOS), and keep the minus sign readable
# 中文字型：依序使用第一個可用的字型 (Linux / Windows / macOS)，並讓負號正常顯示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Microsoft JhengHei', 'PingFang TC',
                                   'Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
# Only petal length and petal width, so the boundaries can be drawn on a plane
# 只用花瓣長度與花瓣寬度，讓決策邊界可以畫在平面上
X = iris.data[:, 2:]
y = iris.target
names = ['山鳶尾 setosa', '變色鳶尾 versicolor', '維吉尼亞鳶尾 virginica']
point_colors = ['red', 'green', 'blue']
region_cmap = ListedColormap(['#ffd6d6', '#d6f5d6', '#d6dcff'])

# Grid covering the plot area; every grid point is classified to color the regions
# 覆蓋整個圖面的網格；對每個網格點分類，用來塗出各區域
xx, yy = np.meshgrid(np.arange(0.5, 7.5, 0.01), np.arange(-0.2, 2.8, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
models = [
    ('KNN (k=1)', KNeighborsClassifier(n_neighbors=1)),
    ('KNN (k=17)', KNeighborsClassifier(n_neighbors=17)),
    ('LDA 線性判別分析', LinearDiscriminantAnalysis()),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
print('模型 | 訓練準確率 | 交叉驗證準確率 (10 折 x 10 次) | 誤判筆數')
for ax, (title, model) in zip(axes.ravel(), models):
    model.fit(X, y)
    region = model.predict(grid).reshape(xx.shape)
    y_pred = model.predict(X)
    cv_scores = cross_val_score(model, X, y, cv=cv)
    wrong = y_pred != y
    print(f'{title} | {(~wrong).mean():.1%} | {cv_scores.mean():.1%} ± {cv_scores.std():.1%} | {wrong.sum()}')

    ax.contourf(xx, yy, region, cmap=region_cmap, alpha=0.9)
    for i, color in enumerate(point_colors):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, edgecolor='k', linewidth=0.4, s=30)
    ax.scatter(X[wrong, 0], X[wrong, 1], marker='x', color='black', s=90, linewidth=2)
    ax.set_title(f'{title}\n訓練準確率 {(~wrong).mean():.1%}｜交叉驗證 {cv_scores.mean():.1%}')

# K-Means does not use labels: map each cluster to its majority species to color the regions
# K-Means 不使用標籤：把每一群對應到群內最多的品種，再依此塗色
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
cluster_to_label = np.array([np.bincount(y[kmeans.labels_ == c]).argmax() for c in range(3)])
region = cluster_to_label[kmeans.predict(grid)].reshape(xx.shape)
y_pred = cluster_to_label[kmeans.labels_]
wrong = y_pred != y
print(f'K-Means (k=3) | 分群純度 {(~wrong).mean():.1%} | 不適用 (非監督式) | {wrong.sum()}')

ax = axes[1, 1]
ax.contourf(xx, yy, region, cmap=region_cmap, alpha=0.9)
for i, color in enumerate(point_colors):
    ax.scatter(X[y == i, 0], X[y == i, 1], color=color, edgecolor='k', linewidth=0.4, s=30)
ax.scatter(X[wrong, 0], X[wrong, 1], marker='x', color='black', s=90, linewidth=2)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', color='gold',
           edgecolor='black', s=400, linewidth=1.2)
ax.set_title(f'K-Means 分群 (k=3，不使用標籤)\n分群純度 {(~wrong).mean():.1%}')

for ax in axes.ravel():
    ax.set_xlabel('花瓣長度 (cm)')
    ax.set_ylabel('花瓣寬度 (cm)')

# One shared legend: region colors, true species, misclassified points, cluster centers
# 共用圖例：區域顏色、實際品種、誤判點、群中心
handles = [Patch(facecolor=region_cmap.colors[i], edgecolor='gray', label=f'判定為 {names[i]} 的區域') for i in range(3)]
handles += [Line2D([], [], marker='o', linestyle='', color=point_colors[i], markeredgecolor='k',
                   label=f'實際品種：{names[i]}') for i in range(3)]
handles += [Line2D([], [], marker='x', linestyle='', color='black', markersize=9, markeredgewidth=2, label='誤判的樣本'),
            Line2D([], [], marker='*', linestyle='', color='gold', markeredgecolor='black', markersize=15,
                   label='K-Means 群中心')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
fig.suptitle('決策邊界比較 (特徵：花瓣長度、花瓣寬度)', fontsize=15)
# Leave room at the bottom for the legend
# 在底部保留圖例的空間
plt.tight_layout(rect=(0, 0.08, 1, 1))

# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: '四種模型在花瓣長度 (X 軸) 與花瓣寬度 (Y 軸) 平面上的決策邊界。底色 = 模型判定的品種區域，圓點 = 實際品種，X = 誤判的樣本。'
       'KNN k=1 的邊界最曲折 (過擬合)，k=17 較平滑；LDA 的邊界是直線；K-Means 不看標籤，邊界是兩個群中心連線的垂直平分線。',
}
# Save every figure to the project's outputs/ folder (keeps results even without a display)
# 將所有圖存到專案的 outputs/ 資料夾 (無圖形介面時也能保留結果)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
for num in plt.get_fignums():
    out_path = OUTPUT_DIR / f'{Path(__file__).stem}_{num}.png'
    plt.figure(num).savefig(out_path, bbox_inches='tight')
    print('已儲存圖檔:', out_path)
    print('  ' + FIGURE_DESCRIPTIONS[num])
plt.show()
