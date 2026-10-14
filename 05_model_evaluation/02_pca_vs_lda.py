# Compare PCA (unsupervised) and LDA (supervised) projections of Iris, and how well KNN classifies each projection
# 比較 PCA (非監督式) 與 LDA (監督式) 對鳶尾花資料的投影，以及 KNN 在投影後的分類準確率

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names
colors = ['red', 'green', 'blue']

# PCA ignores the labels: it keeps the directions of largest total variance
# PCA 不看標籤：保留整體變異最大的方向
pca = PCA().fit(X)
X_pca = pca.transform(X)

# LDA uses the labels: it keeps the directions that best separate the classes (at most 3 - 1 = 2)
# LDA 使用標籤：保留最能分開類別的方向 (最多 3 - 1 = 2 個)
lda = LinearDiscriminantAnalysis().fit(X, y)
X_lda = lda.transform(X)

print('PCA explained variance ratio / PCA 解釋變異比例:', np.round(pca.explained_variance_ratio_, 4))
print('PC1 loadings (sepal L, sepal W, petal L, petal W) / PC1 權重:', np.round(pca.components_[0], 3))
print('LDA explained variance ratio / LDA 類別間差異比例:', np.round(lda.explained_variance_ratio_, 4))

# KNN (k=5) on 1-D and 2-D projections; the projection is fit inside each training fold to avoid leaking test data
# 在一維與二維投影上做 KNN (k=5)；投影放進 pipeline，只用每一折的訓練資料擬合，避免測試資料洩漏
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
print('KNN (k=5) 10-fold x 10 CV accuracy / 交叉驗證準確率:')
for n_dim in (1, 2):
    for name, reducer in [('PCA', PCA(n_components=n_dim)),
                          ('LDA', LinearDiscriminantAnalysis(n_components=n_dim))]:
        scores = cross_val_score(make_pipeline(reducer, KNeighborsClassifier(n_neighbors=5)), X, y, cv=cv)
        print(f'  {n_dim}-D {name}: {scores.mean():.3f} ± {scores.std():.3f}')
scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=cv)
print(f'  original 4-D / 原始 4 維: {scores.mean():.3f} ± {scores.std():.3f}')

# Figure 1: top row = 2-D projections, bottom row = histograms of the first component only
# 圖 1：上排 = 二維投影，下排 = 只看第一個成分的直方圖
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for col, (title, Z, axis_name) in enumerate([
        (f'PCA (unsupervised)', X_pca, 'PC'),
        (f'LDA (supervised)', X_lda, 'LD')]):
    ax = axes[0, col]
    for i, (color, name) in enumerate(zip(colors, names)):
        ax.scatter(Z[y == i, 0], Z[y == i, 1], color=color, alpha=0.7, label=name, facecolor='none')
    ax.set_title(f'{title}: 2-D projection')
    ax.set_xlabel(f'{axis_name}1')
    ax.set_ylabel(f'{axis_name}2')
    ax.legend(loc='best')

    ax = axes[1, col]
    bins = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 31)
    for i, (color, name) in enumerate(zip(colors, names)):
        ax.hist(Z[y == i, 0], bins=bins, color=color, alpha=0.5, label=name)
    ax.set_title(f'{title}: first component only')
    ax.set_xlabel(f'{axis_name}1')
    ax.set_ylabel('Number of samples')
    ax.legend(loc='best')
plt.tight_layout()

# Figure 2: explained variance ratio of each component
# 圖 2：每個成分的解釋變異比例
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (title, ratios, prefix) in zip(axes, [
        ('PCA: share of total variance', pca.explained_variance_ratio_, 'PC'),
        ('LDA: share of between-class variance', lda.explained_variance_ratio_, 'LD')]):
    labels = [f'{prefix}{i + 1}' for i in range(len(ratios))]
    bars = ax.bar(labels, ratios, color='tab:blue')
    ax.bar_label(bars, labels=[f'{r:.1%}' for r in ratios])
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_xlabel('Component')
    ax.set_ylabel('Explained variance ratio')
plt.tight_layout()

# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Top: 2-D projections by PCA (left) and LDA (right). Bottom: histograms of the first component only. Both separate setosa, but on the first component alone LDA keeps versicolor and virginica much further apart than PCA, because LDA uses the labels to choose its direction.',
        '上排：PCA (左) 與 LDA (右) 的二維投影。下排：只看第一個成分的直方圖。兩者都能分出 setosa，但只看第一個成分時，LDA 把 versicolor 與 virginica 分得比 PCA 開得多，因為 LDA 利用標籤選擇投影方向。'),
    2: ('Explained variance ratio. PCA (left): share of the total variance in each principal component. LDA (right): share of the between-class variance in each discriminant. LD1 alone carries 99% of the class separation.',
        '解釋變異比例。PCA (左)：各主成分佔整體變異的比例。LDA (右)：各判別軸佔類別間差異的比例。LD1 單獨就包含 99% 的類別分離資訊。'),
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
