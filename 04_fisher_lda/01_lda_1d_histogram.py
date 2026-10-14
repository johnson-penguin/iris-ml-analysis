# Fisher LDA: project Iris onto one dimension and plot per-class histograms
# Fisher LDA：將鳶尾花投影到一維，畫出各類別的直方圖

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Iris dataset
# 載入鳶尾花資料集
iris = load_iris()
X = iris.data
y = iris.target

# Classify with Fisher Feature One (1-D LDA projection)
# 使用 Fisher Feature One 進行分類
lda = LinearDiscriminantAnalysis(n_components=1)  # keep one feature / 保留一個特徵
X_r1 = lda.fit_transform(X, y)  # fit and project / 訓練並投影

# Plot the histogram of Fisher Feature One
# 繪製 Fisher Feature One 的直方圖
for i in np.unique(y):
    plt.hist(X_r1[y == i], bins=10, label=iris.target_names[i], alpha=0.6)

plt.xlabel('Fisher Feature One')
plt.ylabel('Frequency')
plt.legend(title='Class')
plt.title('Histogram of Fisher Feature One')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Histogram of Fisher Feature One (1-D LDA projection) per species. Setosa is far from the others and versicolor/virginica overlap only slightly, so a single LDA feature already separates the classes well.',
        '各品種在 Fisher Feature One (一維 LDA 投影) 上的直方圖。setosa 與其他兩類距離很遠，versicolor 與 virginica 只有少量重疊，只用一個 LDA 特徵就能把類別分得很好。'),
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
