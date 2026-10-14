# Explore Iris with seaborn pairplots and a sepal scatter plot
# 以 seaborn pairplot 與花萼散佈圖探索鳶尾花資料

# Import libraries and load data
# 匯入相關套件及資料
from sklearn import datasets
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
iris = datasets.load_iris()
data1=pd.DataFrame(np.concatenate((iris.data,iris.target.reshape(150,1)),axis=1),columns=np.append(iris.feature_names,'target'))
data1



data=pd.DataFrame(np.concatenate((iris.data,np.repeat(iris.target_names,50).reshape(150,1)),axis=1),columns=np.append(iris.feature_names,'target'))
data[iris.feature_names]=data[iris.feature_names].apply(pd.to_numeric)
data

# Pairplots of the sepal features and of the petal features
# 花萼特徵與花瓣特徵的 pairplot
sns.pairplot(data.iloc[:,[0,1,4]],hue='target')
sns.pairplot(data.iloc[:,2:5],hue='target')

# Sepal length vs. sepal width, colored by class
# 花萼長度與寬度的散佈圖，依類別上色
plt.figure()
plt.scatter(data1.iloc[:,0],data1.iloc[:,1],c=data1.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: ('Sepal pairplot: sepal length vs. width per species; the diagonal shows distribution curves. Setosa stands apart, while versicolor and virginica overlap heavily.',
        '花萼 pairplot：各品種的花萼長度與寬度；對角線為分佈曲線。setosa 可明顯分開，versicolor 與 virginica 重疊嚴重。'),
    2: ('Petal pairplot: petal length vs. width per species. The three species are almost fully separated, so petal features are the most discriminative.',
        '花瓣 pairplot：各品種的花瓣長度與寬度。三個品種幾乎完全分開，花瓣特徵最具鑑別力。'),
    3: ('Sepal scatter plot colored by class (purple = setosa, teal = versicolor, yellow = virginica); same data as the lower-left panel of figure 1.',
        '花萼散佈圖，依類別上色 (紫 = setosa、青 = versicolor、黃 = virginica)；與圖 1 左下格的資料相同。'),
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


