from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  datasets
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import pandas as pd

iris=datasets.load_iris()

print("整個鳶尾花的資料集維度",iris.data.shape)
iris_x=iris.data
iris_y=iris.target

iris_train_x,iris_test_x,iris_train_y,iris_test_y=train_test_split(iris_x,iris_y,test_size=0.33,random_state=0)


knn=KNeighborsClassifier()
knn.fit(iris_train_x,iris_train_y)
predictresult=knn.predict(iris_test_x)




print("測試資料維度",iris_test_x.shape)
print("實際结果",iris_test_y)
print("預測结果",predictresult)
print("準確率",knn.score(iris_test_x,iris_test_y))


