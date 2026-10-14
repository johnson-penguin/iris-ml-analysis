# Iris ML Analysis

以鳶尾花 (Iris) 資料集練習資料探索、KNN 分類、K-Means 分群與 Fisher 線性判別分析 (LDA)。

## 專案結構

```
iris-ml-analysis/
├── 01_data_exploration/      資料探索
│   ├── 01_dataset_info.py                    印出資料集 keys、特徵名稱、維度
│   ├── 02_train_test_split_scatter_matrix.py 切分訓練/測試集並畫散佈矩陣
│   └── 03_seaborn_pairplot.py                seaborn pairplot 與花萼散佈圖
├── 02_knn_classification/    KNN 分類
│   ├── 01_knn_manual_shuffle_split.py        手動打亂切分 (100 訓練 / 50 測試)
│   └── 02_knn_train_test_split.py            train_test_split 切分並計算準確率
├── 03_kmeans_clustering/     K-Means 分群
│   ├── 01_kmeans_blobs_demo.py               以 make_blobs 隨機資料示範
│   ├── 02_kmeans_iris.py                     Iris 分群結果與原始標籤對照
│   ├── 03_elbow_sse_iris.py                  手肘法 (SSE) 選擇 k，四個特徵
│   └── 04_elbow_sse_from_csv.py              手肘法，讀取 data/iris.csv 單一特徵
├── 04_fisher_lda/            Fisher 線性判別分析
│   ├── 01_lda_1d_histogram.py                投影到一維的直方圖
│   ├── 02_lda_2d_scatter_basic.py            投影到二維的散佈圖 (基本版)
│   ├── 03_lda_2d_scatter_labeled.py          二維散佈圖 + 座標軸標籤
│   ├── 04_lda_2d_scatter_filled.py           二維散佈圖 (實心點)
│   ├── 05_lda_2d_scatter_ellipse.py          二維散佈圖 + 各類別共變異數橢圓
│   └── 06_fisher_manual_histogram.py         以 numpy 手刻 Fisher features
├── data/
│   └── iris.csv              由 sklearn 匯出的 Iris 資料
└── docs/                     報告文件
```

## 執行

所有路徑皆以專案根目錄為基準，可在任何目錄下執行：

```bash
python3 03_kmeans_clustering/04_elbow_sse_from_csv.py
```

需要套件：`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`seaborn`。
