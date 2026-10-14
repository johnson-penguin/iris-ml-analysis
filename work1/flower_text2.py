#鳶尾花資料集
from sklearn.datasets import load_iris
#對集合隨機分為訓練集與測試集，並且return回分好的類別中
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import pandas as pd

#鳶尾花資料集
iris_dataset=load_iris()
iris = datasets.load_iris()



x = iris_dataset['data']
y = iris_dataset['target']

#使用split函數分割訓練集與測試集，這邊我希望用其中50筆資料做測試，而整個資料集共150筆資料，test_size設為0.33
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33,random_state=2)

#測試資料集數量是否符合要求
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
irisdataframe=pd.DataFrame(train_x,columns=iris_dataset.feature_names)



pd.plotting.scatter_matrix(irisdataframe,c=train_y,figsize=(5,5),marker='.',hist_kwds={'bins':20},s=60,alpha=0.5)
plt.show()
