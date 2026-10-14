#导入相关库及数据
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
iris = datasets.load_iris()
data1=pd.DataFrame(np.concatenate((iris.data,iris.target.reshape(150,1)),axis=1),columns=np.append(iris.feature_names,'target'))
data1



data=pd.DataFrame(np.concatenate((iris.data,np.repeat(iris.target_names,50).reshape(150,1)),axis=1),columns=np.append(iris.feature_names,'target'))
data=data.apply(pd.to_numeric,errors='ignore')
data

sns.pairplot(data.iloc[:,[0,1,4]],hue='target')
sns.pairplot(data.iloc[:,2:5],hue='target')

plt.scatter(data1.iloc[:,0],data1.iloc[:,1],c=data1.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()


