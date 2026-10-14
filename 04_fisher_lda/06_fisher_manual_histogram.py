# Compute Fisher features by hand with numpy and plot a histogram of the first one
# 以 numpy 手動計算 Fisher features，並畫出第一個特徵的直方圖

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.patches import Ellipse

# Load the Iris dataset
# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target


# Compute Fisher features
# 計算 Fisher features
def fisher_features(X, y):
    # Mean vector of each class
    # 計算每個類別的均值向量
    means = []
    for i in np.unique(y):
        means.append(np.mean(X[y == i], axis=0))

    # Between-class scatter matrix
    # 類別間散佈矩陣
    Sb = np.zeros((X.shape[1], X.shape[1]))
    overall_mean = np.mean(X, axis=0)
    for i, mean_vec in enumerate(means):
        n = X[y == i].shape[0]
        mean_vec = mean_vec.reshape(-1, 1)
        overall_mean = overall_mean.reshape(-1, 1)
        Sb += n * (mean_vec - overall_mean) @ (mean_vec - overall_mean).T

    # Within-class scatter matrix
    # 類別內散佈矩陣
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for i, mean_vec in enumerate(means):
        cov_mat = np.cov(X[y == i].T)
        Sw += cov_mat

    # Fisher features: top eigenvectors of inv(Sw) @ Sb
    # Fisher features：inv(Sw) @ Sb 的前幾大特徵向量
    eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    idx = np.argsort(eigenvals)[::-1]
    W = eigenvecs[:, idx[:2]]  # take the top two eigenvectors / 選取前兩個特徵向量

    return W


# Project the data into the Fisher feature space (matrix multiplication)
# 將資料投影到 Fisher features 空間中 (矩陣乘法)
W = fisher_features(X, y)
w = -fisher_features(X, y)

X_fisher = X @ W
x_fisher = X @ w

# Negate the Fisher feature values (w = -W above)
# 對 Fisher features 的值取負值 (即上方的 w = -W)


# Plot the Fisher feature classification result
# 繪製 Fisher features 分類結果圖
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']
labels = ['Setosa', 'Versicolor', 'Virginica']

for i in range(3):
    plt.hist(x_fisher[y == i, 0], density=False, color='white',edgecolor = colors[i], cumulative = False, label = labels[i])

plt.xlabel('Fisher Feature One')
plt.ylabel('Frequency')
plt.title('Fisher Features - Iris Dataset')
plt.legend()

# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Histogram of the first Fisher feature computed by hand with numpy (sign flipped), per species. It should match 01_lda_1d_histogram up to sign and scale.',
        '以 numpy 手動計算的第一個 Fisher feature (取負值) 在各品種的直方圖。結果應與 01_lda_1d_histogram 相同，只差正負號與比例。'),
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