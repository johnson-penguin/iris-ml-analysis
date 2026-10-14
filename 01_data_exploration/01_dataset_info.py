# Print basic information about the Iris dataset (keys, feature names, shape)
# 印出鳶尾花資料集的基本資訊 (keys、特徵名稱、資料維度)

from sklearn import  datasets

# Load the Iris dataset: a dict-like object holding 150 flower samples
# 載入鳶尾花資料集，以字典形式記錄 150 筆鳶尾花的資料
iris_dataset = datasets.load_iris()

print("資料集的keys",iris_dataset.keys())
print("特徵名稱",iris_dataset['feature_names'])
# 150 samples in total, each with 4 features
# 一共 150 筆資料，每筆資料有四種特徵
print("資料維度",iris_dataset['data'].shape)
