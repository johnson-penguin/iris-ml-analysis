# Fisher LDA: 2-D scatter of the Iris projection with a covariance ellipse per class
# Fisher LDA：鳶尾花投影到二維的散佈圖，並為每個類別畫出共變異數橢圓

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from matplotlib.patches import Ellipse


iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Project the data onto two Fisher features
# 將資料投影到兩個 Fisher features
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

plt.figure()
colors = ['red', 'green', 'blue']
lw = 2

# Plot each class in its own color
# 每個類別用不同顏色畫出
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name, marker='o', facecolor='none'
    )

# Draw an ellipse from each class's mean and covariance (axes = 2 * sqrt(eigenvalues))
# 以各類別的平均值與共變異數畫出橢圓 (軸長 = 2 * sqrt(特徵值))
for i in range(3):
    colors = ['red', 'green','blue']
    labels = ['Setosa', 'Versicolor', 'Virginica']
    mean_vec = np.mean(X_r2[y == i], axis=0)
    cov_mat = np.cov(X_r2[y == i].T)
    eigenvals, eigenvecs = np.linalg.eig(cov_mat)
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvals)
    ellipse = Ellipse(xy=mean_vec, width=width, height=height, angle=angle, alpha=0.5,facecolor="none", edgecolor=colors[i])
    plt.gca().add_patch(ellipse)



plt.legend(loc="best", shadow=False, scatterpoints=1)

plt.title("LDA of IRIS dataset")
plt.xlabel('Fisher Feature One')
plt.ylabel('Fisher Feature Two')
plt.title('Fisher Features - Iris Dataset')
plt.legend()
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('2-D LDA projection with one ellipse per species, drawn from its mean and covariance. The ellipses show each class\'s spread and orientation; ellipses that do not overlap mean well-separated classes.',
        '二維 LDA 投影，並以各品種的平均值與共變異數畫出橢圓。橢圓表示每個類別的分散程度與方向；橢圓不重疊代表類別分得很開。'),
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