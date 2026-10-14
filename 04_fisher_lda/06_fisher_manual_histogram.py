import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.patches import Ellipse

# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target


# 計算 Fisher features
def fisher_features(X, y):
    # 計算每個類別的均值向量
    means = []
    for i in np.unique(y):
        means.append(np.mean(X[y == i], axis=0))

    # between-class scatter matrix
    Sb = np.zeros((X.shape[1], X.shape[1]))
    overall_mean = np.mean(X, axis=0)
    for i, mean_vec in enumerate(means):
        n = X[y == i].shape[0]
        mean_vec = mean_vec.reshape(-1, 1)
        overall_mean = overall_mean.reshape(-1, 1)
        Sb += n * (mean_vec - overall_mean) @ (mean_vec - overall_mean).T

    # within-class scatter matrix
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for i, mean_vec in enumerate(means):
        cov_mat = np.cov(X[y == i].T)
        Sw += cov_mat

    # Fisher features
    eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    idx = np.argsort(eigenvals)[::-1]
    W = eigenvecs[:, idx[:2]]  # 選取前兩個特徵向量

    return W


# 將資料投影到 Fisher features 空間中，矩陣乘法
W = fisher_features(X, y)
w = -fisher_features(X, y)

X_fisher = X @ W
x_fisher = X @ w

# 對 Fisher features 的值取負值


# 繪製 Fisher features 分類結果圖
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']
labels = ['Setosa', 'Versicolor', 'Virginica']

for i in range(3):
    plt.hist(x_fisher[y == i, 0], density=False, color='white',edgecolor = colors[i], cumulative = False, label = labels[i])

plt.xlabel('Fisher Feature One')
plt.ylabel('Fisher Feature Two')
plt.title('Fisher Features - Iris Dataset')
plt.legend()

plt.show()