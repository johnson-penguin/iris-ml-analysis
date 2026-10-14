import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()

X = iris ['data']   #150個樣本，一共四個屬性(花萼長、花萼寬、花瓣長度、花瓣寬度)
Y = iris ['target']
#分兩組數據 訓練用、測試用
#順序打亂

#原始順序
index = np.arange(150)
#亂數打亂
np.random.shuffle(index)
#亂數打亂後的前100個用作knn訓練，後50個用作模型測試
X_train , X_text = X[index[:100]] , X[index[100:]]
Y_train , Y_text = Y[index[:100]] , Y[index[100:]]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
Y_ = knn.predict(X_text)

#功能對比
print(Y_)
print('-----------------------')
print(Y_text)