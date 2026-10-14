import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from matplotlib.patches import Ellipse


iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

plt.figure()
colors = ['red', 'green', 'blue']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name, marker='o', facecolor='none'
    )

for i in range(3):
    colors = ['red', 'green','blue']
    labels = ['Setosa', 'Versicolor', 'Virginica']
    mean_vec = np.mean(X_r2[y == i], axis=0)
    cov_mat = np.cov(X_r2[y == i].T)
    eigenvals, eigenvecs = np.linalg.eig(cov_mat)
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvals)
    ellipse = Ellipse(xy=mean_vec, width=width, height=height, angle=angle, alpha=0.5,facecolor="none", edgecolor=colors[i])
    plt.gca().add_patch(ellipse)



plt.legend(loc="best", shadow=False, scatterpoints=1)

plt.title("LDA of IRIS dataset")
plt.xlabel('Fisher Feature One')
plt.ylabel('Fisher Feature Two')
plt.title('Fisher Features - Iris Dataset')
plt.legend()
plt.show()