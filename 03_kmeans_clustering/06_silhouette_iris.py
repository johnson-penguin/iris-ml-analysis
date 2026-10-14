# Silhouette analysis: choose k for K-Means by the average silhouette score on all four Iris features
# 輪廓分析：以平均輪廓係數選擇 K-Means 的 k 值 (使用鳶尾花全部四個特徵)

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

dx = load_iris().data

# Silhouette needs at least 2 clusters, so k starts from 2
# 輪廓係數至少需要 2 群，所以 k 從 2 開始
K_RANGE = range(2, 9)

# Average silhouette score of each k: close to 1 = compact and well separated, close to 0 = overlapping clusters
# 每個 k 的平均輪廓係數：越接近 1 表示群內緊密且群間分離，接近 0 表示群與群重疊
scores = []
for k in K_RANGE:
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(dx)
    scores.append(silhouette_score(dx, labels))
    print(f'k={k}  silhouette={scores[-1]:.3f}')

best_k = K_RANGE[scores.index(max(scores))]
print('Best k / 最佳 k:', best_k)

plt.plot(K_RANGE, scores, 'o-')
plt.xlabel('k')
plt.ylabel('Average silhouette score')
plt.title('Silhouette analysis - Iris (4 features)')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: (f'Average silhouette score for k = 2-8 using all 4 features. The highest score is the best k (here k = {best_k}); the score generally falls as k grows because clusters start to overlap.',
        f'使用全部 4 個特徵，k = 2-8 時的平均輪廓係數。分數最高者為最佳 k (此處 k = {best_k})；k 越大，群與群越重疊，分數大致隨之下降。'),
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
