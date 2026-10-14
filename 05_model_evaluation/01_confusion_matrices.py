# Confusion matrices of KNN, LDA and K-Means on Iris, with per-class precision / recall
# 比較 KNN、LDA、K-Means 在鳶尾花資料上的混淆矩陣，並列出各類別的 precision / recall

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

# Best k found by 02_knn_classification/03_knn_cross_validation.py
# 由 02_knn_classification/03_knn_cross_validation.py 找出的最佳 k
BEST_K = 17

# Supervised models: every sample is predicted by a model that never saw it (10-fold cross_val_predict)
# 監督式模型：以 10-fold cross_val_predict 預測，每筆樣本都由沒看過它的模型預測
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
predictions = {
    f'KNN (k={BEST_K}, 10-fold CV)': cross_val_predict(KNeighborsClassifier(n_neighbors=BEST_K), X, y, cv=cv),
    'LDA (10-fold CV)': cross_val_predict(LinearDiscriminantAnalysis(), X, y, cv=cv),
}

# K-Means is unsupervised: cluster all data, then map each cluster to its majority species
# K-Means 是非監督式：對全部資料分群後，把每一群對應到群內最多的品種
clusters = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
cluster_to_label = {c: np.bincount(y[clusters == c]).argmax() for c in np.unique(clusters)}
predictions['K-Means (k=3, majority mapping)'] = np.array([cluster_to_label[c] for c in clusters])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (title, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{title}\naccuracy = {accuracy_score(y, y_pred):.1%}')
    print(f'===== {title} =====')
    print(classification_report(y, y_pred, target_names=names, digits=3))

plt.tight_layout()
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Confusion matrices of KNN, LDA (both 10-fold cross-validated) and K-Means (k = 3). Rows = true species, columns = predicted species; the diagonal counts correct predictions. All errors fall between versicolor and virginica; setosa is never confused.',
        'KNN、LDA (皆為 10-fold 交叉驗證) 與 K-Means (k = 3) 的混淆矩陣。列 = 實際品種，欄 = 預測品種；對角線為預測正確的筆數。所有錯誤都發生在 versicolor 與 virginica 之間，setosa 從未被混淆。'),
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
