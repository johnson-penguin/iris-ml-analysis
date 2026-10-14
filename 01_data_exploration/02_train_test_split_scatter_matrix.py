# Split Iris into training/test sets and plot a scatter matrix of the training set
# 將鳶尾花資料切分為訓練集與測試集，並畫出訓練集的散佈矩陣

# Iris dataset
# 鳶尾花資料集
from sklearn.datasets import load_iris
# Randomly split the data into training and test sets
# 將資料隨機分為訓練集與測試集，並回傳切分好的結果
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Iris dataset
# 鳶尾花資料集
iris_dataset=load_iris()
iris = datasets.load_iris()



x = iris_dataset['data']
y = iris_dataset['target']

# Split with train_test_split: use 50 of the 150 samples for testing, so test_size=0.33
# 使用 train_test_split 分割訓練集與測試集：150 筆資料中取 50 筆做測試，所以 test_size 設為 0.33
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33,random_state=2)

# Check that the split sizes are as expected
# 檢查資料集筆數是否符合要求
print("原始資料X筆數:", x.shape[0],"訓練用資料筆數:", train_x.shape[0],"測試用資料筆數:", test_x.shape[0])
print("原始資料y筆數:", y.shape[0],"訓練用資料筆數:", train_y.shape[0],"測試用資料筆數:", test_y.shape[0])


'''
print('trainx\n',train_x)
print('trainy\n',train_y)
print('testx\n',test_x)
print('testy\n',test_y)
print(test_x.shape)
print(test_x.shape)
'''
# Scatter matrix of the training set, colored by class
# 訓練集的散佈矩陣，依類別上色
irisdataframe=pd.DataFrame(train_x,columns=iris_dataset.feature_names)



pd.plotting.scatter_matrix(irisdataframe,c=train_y,figsize=(5,5),marker='.',hist_kwds={'bins':20},s=60,alpha=0.5)
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Scatter matrix of the training set: every pair of the 4 features, colored by class; the diagonal shows each feature\'s histogram. Petal features separate the classes best.',
        '訓練集的散佈矩陣：4 個特徵兩兩配對，顏色代表類別；對角線為各特徵的直方圖。花瓣特徵最能區分類別。'),
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
