# K-Means clustering on Iris, compared with the original labels
# 對鳶尾花資料做 K-Means 分群，並與原始標籤對照

'''
import platform
import psutil

print('執行緒:',psutil.cpu_count(),'核')
print('記憶體:',round((psutil.virtual_memory().total/(1024 ** 3))),'GB')

print(platform.platform())
'''
# Iris dataset
# 鳶尾花資料集
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import datasets
from pathlib import Path
import matplotlib.pyplot as plt
'''
n1 = float(input('請輸入分群數量 : ')) // 1
'''
# Number of clusters
# 分群數量
n1 = int(8)

# Iris dataset
# 鳶尾花資料集
iris_dataset=load_iris()
iris = datasets.load_iris()

dx = iris_dataset['data']
dy = iris_dataset['target']
# Use KMeans to find n clusters in the data
# 用 KMeans 在資料中找出 n 個分組
kmeans = KMeans(n_clusters=n1)
kmeans.fit(dx)

# Predict the new cluster labels
# 預測新的目標值 label
new_dy = kmeans.predict(dx)





# Plot the original data
# 畫出原始的群資料

plt.rcParams['font.size'] = 8
plt.figure(figsize=(10,5))

plt.subplot(221)  # 2x2 grid, position 1 / 2 列 2 欄，目前位置為 1
plt.title('Original Sepal length VS Sepal width')
plt.scatter(dx.T[0], dx.T[1], c=dy, cmap=plt.cm.Set1)

plt.subplot(222)  # 2x2 grid, position 2 / 2 列 2 欄，目前位置為 2
plt.title('Original Sepal length VS Petal length')
plt.scatter(dx.T[0], dx.T[2], c=dy, cmap=plt.cm.Set1)


# Plot the data colored by the n KMeans clusters
# 根據重新分成的 n 組來畫出資料
n1 = str(n1)
plt.subplot(223)
plt.title(n1 + ' groups Sepal length VS Sepal width')
plt.scatter(dx.T[0], dx.T[1], c=new_dy, cmap=plt.cm.Set1)

plt.subplot(224)
plt.title(n1 + ' groups Sepal length VS Petal length')
plt.scatter(dx.T[0], dx.T[2], c=new_dy, cmap=plt.cm.Set1)




# Show the figure
# 顯示圖表
plt.tight_layout()
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: (f'Top row: original species labels; bottom row: the {n1} KMeans clusters. Left column: sepal length vs. sepal width; right column: sepal length vs. petal length. Compare top and bottom to see how the clusters map to the real species.',
        f'上排：原始品種標籤；下排：KMeans 分成的 {n1} 群。左欄：花萼長度 vs 花萼寬度；右欄：花萼長度 vs 花瓣長度。上下對照可看出分群與實際品種的對應。'),
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

