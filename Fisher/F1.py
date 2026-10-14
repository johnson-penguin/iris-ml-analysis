import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 載入鳶尾花資料集
iris = load_iris()
X = iris.data
y = iris.target

# 使用Fisher Feature One進行分類
lda = LinearDiscriminantAnalysis(n_components=1) #feature
X_r1 = lda.fit_transform(X, y)#做

# 繪製Fisher Feature One的直方圖
for i in np.unique(y):
    plt.hist(X_r1[y == i], bins=10, label=iris.target_names[i], alpha=0.6)

plt.xlabel('Fisher Feature One')
plt.ylabel('Frequency')
plt.legend(title='Class')
plt.title('Histogram of Fisher Feature One')
plt.show()
