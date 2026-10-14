# KNN classification with train_test_split, reporting test accuracy
# 以 train_test_split 切分資料進行 KNN 分類，並計算測試準確率

from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  datasets
import matplotlib.pyplot as plt
import pandas as pd

iris=datasets.load_iris()

print("整個鳶尾花的資料集維度",iris.data.shape)
iris_x=iris.data
iris_y=iris.target

# Use 2/3 of the data for training and 1/3 for testing
# 三分之二資料做訓練、三分之一做測試
iris_train_x,iris_test_x,iris_train_y,iris_test_y=train_test_split(iris_x,iris_y,test_size=0.33,random_state=0)


# Train a KNN classifier (default k=5) and predict the test set
# 訓練 KNN 分類器 (預設 k=5) 並預測測試集
knn=KNeighborsClassifier()
knn.fit(iris_train_x,iris_train_y)
predictresult=knn.predict(iris_test_x)




print("測試資料維度",iris_test_x.shape)
print("實際结果",iris_test_y)
print("預測结果",predictresult)
print("準確率",knn.score(iris_test_x,iris_test_y))


