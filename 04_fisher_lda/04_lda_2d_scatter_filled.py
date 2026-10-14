# Fisher LDA: 2-D scatter of the Iris projection, with filled markers
# Fisher LDA：鳶尾花投影到二維的散佈圖 (實心點)

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Iris dataset
# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target

# Linear discriminant analysis with Fisher features
# 使用 Fisher 特徵進行線性判別分析
lda = LinearDiscriminantAnalysis(n_components=2)
X_fisher = lda.fit_transform(X, y)
# Plot the classification result
# 繪製分類結果
colors = ['navy', 'turquoise', 'darkorange']
target_names = iris.target_names
plt.figure()


for color,i,target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_fisher[y == i,0],X_fisher[y == i, 1],color=color,alpha = 0.8,lw=2,label= target_name)



plt. legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Fisher Feature One vs. Fisher Feature Two')
plt.xlabel('Fisher Feature One')
plt.ylabel('Fisher Feature Two')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('2-D LDA projection drawn with filled markers. The classes separate mainly along Fisher Feature One.',
        '以實心點畫出的二維 LDA 投影。類別主要沿著 Fisher Feature One 分開。'),
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