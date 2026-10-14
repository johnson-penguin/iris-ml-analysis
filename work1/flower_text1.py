from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  datasets
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import pandas as pd

iris_dataset = datasets.load_iris()#鳶尾花資料集，字典形式紀錄150比鳶尾花的資料

print("資料集的keys",iris_dataset.keys())
print("特徵名稱",iris_dataset['feature_names'])
print("資料維度",iris_dataset['data'].shape)#一共150比資料，並且每筆資料擁有四種特徵
