# Fisher LDA: project Iris onto two dimensions and plot a scatter (basic)
# Fisher LDA：將鳶尾花投影到二維並畫出散佈圖 (基本版)

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")

# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('2-D LDA projection (Fisher Feature One vs. Two), one color per species. The classes separate mainly along Feature One (x-axis).',
        '二維 LDA 投影 (Fisher Feature One vs Two)，每個品種一種顏色。類別主要沿著 Feature One (x 軸) 分開。'),
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