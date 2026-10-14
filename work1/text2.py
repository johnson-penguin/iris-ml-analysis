'''
import platform
import psutil

print('執行緒:',psutil.cpu_count(),'核')
print('記憶體:',round((psutil.virtual_memory().total/(1024 ** 3))),'GB')

print(platform.platform())
'''
#鳶尾花資料集
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
'''
n1 = float(input('請輸入分群數量 : ')) // 1
'''
n1 = int(8)

#鳶尾花資料集
iris_dataset=load_iris()
iris = datasets.load_iris()

dx = iris_dataset['data']
dy = iris_dataset['target']
# 用 KMeans 在資料中找出 n 個分組
kmeans = KMeans(n_clusters=n1)
kmeans.fit(dx)

# 預測新的目標值 label
new_dy = kmeans.predict(dx)





# 畫出原始的群資料

plt.rcParams['font.size'] = 8
plt.figure(figsize=(10,5))

plt.subplot(221)#2行2列, 当前位置为1
plt.title('Original Sepal length VS Sepal width')
plt.scatter(dx.T[0], dx.T[1], c=dy, cmap=plt.cm.Set1)

plt.subplot(222)#2行列, 当前位置为1
plt.title('Original Petal length VS Petal width ')
plt.scatter(dx.T[0], dx.T[2], c=dy, cmap=plt.cm.Set1)


# 根據重新分成的 n 組來畫出資料
n1 = str(n1)
plt.subplot(223)
plt.title(n1 + ' Sepal length VS Sepal width')
plt.scatter(dx.T[0], dx.T[1], c=new_dy, cmap=plt.cm.Set1)

plt.subplot(224)
plt.title(n1 + ' groups Petal length VS Petal width ')
plt.scatter(dx.T[0], dx.T[2], c=new_dy, cmap=plt.cm.Set1)




# 顯示圖表
plt.tight_layout()
plt.show()

