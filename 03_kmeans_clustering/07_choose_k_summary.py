# Choose k by combining every metric (internal and external) on three feature sets: all 4, sepal only, petal only
# 綜合所有指標 (內部與外部) 與三種特徵組合 (全部 4 個、只用花萼、只用花瓣) 來選擇 k

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)

X, y = load_iris(return_X_y=True)
K_RANGE = list(range(2, 9))
FEATURE_SETS = [
    ('All 4 features', X),
    ('Sepal only', X[:, :2]),
    ('Petal only', X[:, 2:]),
]

# Metrics: (name, higher is better?, uses labels?)
# Internal metrics only look at the data; external metrics compare clusters with the true species
# 指標：(名稱, 是否越大越好, 是否使用標籤)
# 內部指標只看資料本身；外部指標把分群結果與實際品種比較
METRICS = [
    ('Silhouette', True, False),
    ('Calinski-Harabasz', True, False),
    ('Davies-Bouldin', False, False),
    ('ARI', True, True),
    ('NMI', True, True),
]

results = {}  # results[feature set][metric] = list of scores for each k / 各 k 的分數
for set_name, data in FEATURE_SETS:
    scores = {m[0]: [] for m in METRICS}
    scores['SSE'] = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(data)
        labels = kmeans.labels_
        scores['SSE'].append(kmeans.inertia_)
        scores['Silhouette'].append(silhouette_score(data, labels))
        scores['Calinski-Harabasz'].append(calinski_harabasz_score(data, labels))
        scores['Davies-Bouldin'].append(davies_bouldin_score(data, labels))
        scores['ARI'].append(adjusted_rand_score(y, labels))
        scores['NMI'].append(normalized_mutual_info_score(y, labels))
    results[set_name] = scores

# Best k of each metric on each feature set, and how many times each k wins
# 每個指標在每種特徵組合下的最佳 k，以及每個 k 勝出的次數
print('Best k per metric / 各指標的最佳 k')
print(f'{"metric":<20}' + ''.join(f'{s:>16}' for s, _ in FEATURE_SETS))
votes = {'internal': {k: 0 for k in K_RANGE}, 'external': {k: 0 for k in K_RANGE}}
best_k = {}
for name, higher, external in METRICS:
    row = []
    for set_name, _ in FEATURE_SETS:
        values = np.array(results[set_name][name])
        k = K_RANGE[int(values.argmax() if higher else values.argmin())]
        best_k[(name, set_name)] = k
        votes['external' if external else 'internal'][k] += 1
        row.append(k)
    print(f'{name:<20}' + ''.join(f'{k:>16}' for k in row))
print('Votes / 得票數 (internal = 不用標籤, external = 用標籤):')
for kind in ('internal', 'external'):
    print(f'  {kind:<9}', {k: v for k, v in votes[kind].items() if v})

# Figure: one panel per feature set; metrics on a 0-1 scale as lines, the best k of each metric as a big marker
# 圖：每種特徵組合一張子圖；0-1 範圍的指標畫成折線，各指標的最佳 k 用大圓點標出
styles = {'Silhouette': ('tab:orange', 'o--'), 'ARI': ('tab:blue', 's-'), 'NMI': ('tab:green', '^-')}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, (set_name, _) in zip(axes, FEATURE_SETS):
    for name, (color, fmt) in styles.items():
        values = results[set_name][name]
        ax.plot(K_RANGE, values, fmt, color=color, label=name)
        k = best_k[(name, set_name)]
        ax.plot(k, values[K_RANGE.index(k)], 'o', color=color, markersize=14, markerfacecolor='none', markeredgewidth=2)
    ax.axvline(3, color='gray', linestyle=':', label='k = 3 (number of species)')
    ax.set_title(set_name)
    ax.set_xlabel('k (number of clusters)')
    ax.set_xticks(K_RANGE)
axes[0].set_ylabel('Score (higher is better)')
axes[0].legend(loc='lower left', fontsize=9)
fig.suptitle('Choosing k: internal metric (Silhouette, dashed) vs. external metrics with labels (ARI, NMI, solid); circles = best k')
plt.tight_layout()

# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Silhouette (dashed, no labels) and ARI / NMI (solid, compared with the true species) for k = 2-8 on three feature sets; circles mark each metric\'s best k. Silhouette always prefers k = 2, while ARI and NMI peak at k = 3 on every feature set.',
        '三種特徵組合下 k = 2-8 的輪廓係數 (虛線，不用標籤) 與 ARI / NMI (實線，與實際品種比較)；圓圈為各指標的最佳 k。輪廓係數一律偏好 k = 2，ARI 與 NMI 則在每種特徵組合下都於 k = 3 達到最高。'),
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
