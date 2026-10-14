import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#入Iris數據集
iris = load_iris()
X = iris.data
y = iris.target

#使用Fisher特徵進行線性判刷分析
lda = LinearDiscriminantAnalysis(n_components=2)
X_fisher = lda.fit_transform(X, y)
#繪製分類結果
colors = ['navy', 'turquoise', 'darkorange']
target_names = iris.target_names
plt.figure()


for color,i,target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(X_fisher[y == i,0],X_fisher[y == i, 1],color=color,alpha = 0.8,lw=2,label= target_name)



plt. legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Fisher Feature One vs. Fisher Feature Two')
plt.xlabel('Fisher Feature One')
plt.ylabel('Fisher Feature Two')
plt.show()