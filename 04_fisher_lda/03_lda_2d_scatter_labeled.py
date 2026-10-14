# Fisher LDA: 2-D scatter of the Iris projection, with axis labels
# Fisher LDA：鳶尾花投影到二維的散佈圖，加上座標軸標籤

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

lda = LinearDiscriminantAnalysis(n_components=2)
# Project the data into the Fisher feature space (matrix multiplication)
# 將資料投影到 Fisher features 空間中 (矩陣乘法)
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
plt.xlabel('Fisher Feature One')
plt.ylabel('Fisher Feature Two')
plt.title('Fisher Features - Iris Dataset')
plt.legend()
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('2-D LDA projection with axis labels (x = Fisher Feature One, y = Fisher Feature Two). The classes separate mainly along Feature One.',
        '加上座標軸標籤的二維 LDA 投影 (x = Fisher Feature One、y = Fisher Feature Two)。類別主要沿著 Feature One 分開。'),
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