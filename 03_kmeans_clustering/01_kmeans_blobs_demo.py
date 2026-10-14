import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

'''
n1 = float(input('請輸入分群數量 : ')) // 1
'''
n1 = int(2)

'''
print('將資料分為' ,n1,'群')

'''

# 隨機產生 c 組 b features 的資料 a 筆 (dy 即原始目標值 label 0~9)
# dx, dy = make_blobs(n_samples=a, n_features=b, centers=c, shuffle='True')
dx, dy = make_blobs(n_samples=80, n_features=2, centers=1, shuffle=True)

# 用 KMeans 在資料中找出 n 個分組
kmeans = KMeans(n_clusters=n1)
kmeans.fit(dx)

# 預測新的目標值 label
new_dy = kmeans.predict(dx)
plt.rcParams['font.size'] = 14
plt.figure(figsize=(16, 8))

# 畫出原始的群資料
plt.subplot(121)
plt.title('Original data ')
plt.scatter(dx.T[0], dx.T[1], c=dy, cmap=plt.cm.Set1)

# 根據重新分成的 n 組來畫出資料
n1 = str(n1)
plt.subplot(122)
plt.title('KMeans=' + n1 + ' groups')
plt.scatter(dx.T[0], dx.T[1], c=new_dy, cmap=plt.cm.Set1)

# 顯示圖表
plt.tight_layout()
plt.show()