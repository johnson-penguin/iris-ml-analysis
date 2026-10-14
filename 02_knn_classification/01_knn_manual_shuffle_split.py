# KNN classification with a manual shuffle split (100 train / 50 test)
# 手動打亂並切分資料 (100 筆訓練 / 50 筆測試) 進行 KNN 分類

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()

# 150 samples with 4 features (sepal length, sepal width, petal length, petal width)
# 150 個樣本，共四個屬性 (花萼長度、花萼寬度、花瓣長度、花瓣寬度)
X = iris ['data']
Y = iris ['target']
# Split into two groups: training and test
# 分成兩組資料：訓練用、測試用
# Shuffle the order
# 打亂順序

# Original order
# 原始順序
index = np.arange(150)
# Shuffle randomly
# 亂數打亂
np.random.shuffle(index)
# The first 100 shuffled samples train the KNN, the last 50 test the model
# 打亂後的前 100 筆用於 KNN 訓練，後 50 筆用於模型測試
X_train , X_text = X[index[:100]] , X[index[100:]]
Y_train , Y_text = Y[index[:100]] , Y[index[100:]]

# Train a KNN classifier with k=5 and predict the test set
# 以 k=5 訓練 KNN 分類器並預測測試集
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
Y_ = knn.predict(X_text)

# Compare the predictions with the true labels
# 比對預測結果與實際標籤
print(Y_)
print('-----------------------')
print(Y_text)