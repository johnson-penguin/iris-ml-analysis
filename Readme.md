# Iris ML Analysis

以鳶尾花 (Iris) 資料集練習資料探索、KNN 分類、K-Means 分群與 Fisher 線性判別分析 (LDA)。

每張圖的 X/Y 軸意義與觀察結果請見 **[分析報告 docs/Report.md](docs/Report.md)**，重點整理見 [重點結論](docs/Report.md#2-重點結論)。

## 專案結構

```
iris-ml-analysis/
├── 01_data_exploration/      資料探索
├── 02_knn_classification/    KNN 分類
├── 03_kmeans_clustering/     K-Means 分群
├── 04_fisher_lda/            Fisher 線性判別分析
├── 05_model_evaluation/      模型評估與比較
├── data/iris.csv             由 sklearn 匯出的 Iris 資料 (150 筆，4 個特徵 + target)
├── docs/
│   ├── Report.md             分析報告：每張圖的說明與觀察
│   └── figures/              報告使用的圖檔
├── outputs/                  執行後產生的圖 (不納入 git)
└── requirements.txt          固定版本的套件清單
```

## 各程式功能

「輸出」欄中的圖存在 `outputs/<檔名>_<編號>.png`，存檔時終端機會印出路徑與這張圖的意義。

### 01_data_exploration — 資料探索

報告：[3. 資料探索](docs/Report.md#3-資料探索)

| 程式 | 功能 | 輸出 |
|---|---|---|
| `01_dataset_info.py` | 印出資料集的 keys、4 個特徵名稱、資料維度 (150, 4) | 文字 |
| `02_train_test_split_scatter_matrix.py` | 以 `train_test_split` 切成 100 筆訓練 / 50 筆測試，畫出訓練集的 4×4 散佈矩陣 | 文字 + 1 張圖 |
| `03_seaborn_pairplot.py` | 用 seaborn 分別畫花萼、花瓣兩個特徵的 pairplot，再畫一張花萼長度 vs 寬度散佈圖 | 3 張圖 |

### 02_knn_classification — KNN 分類

報告：[4. KNN 分類](docs/Report.md#4-knn-分類)

| 程式 | 功能 | 輸出 |
|---|---|---|
| `01_knn_manual_shuffle_split.py` | 用 numpy 手動打亂資料，前 100 筆訓練、後 50 筆測試，以 k=5 的 KNN 預測並印出預測值與實際值 (固定亂數種子) | 文字 |
| `02_knn_train_test_split.py` | 用 `train_test_split` (random_state=0) 切分，以 KNN 預測並計算準確率 (0.98) | 文字 |
| `03_knn_cross_validation.py` | 以重複分層 10-fold 交叉驗證 (10 折 × 10 次) 比較 k=1–30 的訓練與驗證準確率，找出最佳 k (17) 並觀察過擬合/欠擬合 | 文字 + 1 張圖 |

### 03_kmeans_clustering — K-Means 分群

報告：[5. K-Means 分群](docs/Report.md#5-k-means-分群)

| 程式 | 功能 | 輸出 |
|---|---|---|
| `01_kmeans_blobs_demo.py` | 以 `make_blobs` 產生單一群的隨機資料，示範 K-Means 仍會切成 k 群 | 1 張圖 |
| `02_kmeans_iris.py` | 對 Iris 做 K-Means (k=8)，以 2×2 子圖對照原始品種與分群結果 | 1 張圖 |
| `03_elbow_sse_iris.py` | 手肘法：用全部 4 個特徵計算 k=1–8 的 SSE，找出合適的 k | 1 張圖 |
| `04_elbow_sse_from_csv.py` | 手肘法：讀取 `data/iris.csv`，只用 sepal length 計算 k=1–8 的 SSE | 1 張圖 |
| `05_kmeans_compare_k.py` | 比較 k = 2、3、4、6、8 的分群結果：只用花萼特徵 vs 只用花瓣特徵，並印出各自的純度；另畫出 k = 3 時兩者的誤判樣本 | 文字 + 2 張圖 |
| `06_silhouette_iris.py` | 輪廓分析：用全部 4 個特徵計算 k=2–8 的平均輪廓係數，找出最佳 k | 文字 + 1 張圖 |

### 04_fisher_lda — Fisher 線性判別分析

報告：[6. Fisher 線性判別分析 (LDA)](docs/Report.md#6-fisher-線性判別分析-lda)

| 程式 | 功能 | 輸出 |
|---|---|---|
| `01_lda_1d_histogram.py` | 用 sklearn LDA 投影到一維，畫出各品種在 Fisher Feature One 上的直方圖 | 1 張圖 |
| `02_lda_2d_scatter_basic.py` | LDA 投影到二維，畫出空心點散佈圖 (無軸標籤) | 1 張圖 |
| `03_lda_2d_scatter_labeled.py` | 同上，加上 Fisher Feature One / Two 軸標籤 | 1 張圖 |
| `04_lda_2d_scatter_filled.py` | 同上，改用實心點與不同配色 | 1 張圖 |
| `05_lda_2d_scatter_ellipse.py` | 二維散佈圖，並以各品種的平均值與共變異數畫出 ±1 標準差橢圓 | 1 張圖 |
| `06_fisher_manual_histogram.py` | 不用 sklearn，以 numpy 手動計算 inv(Sw)·Sb 的特徵向量得到 Fisher features，畫出直方圖 | 1 張圖 |

### 05_model_evaluation — 模型評估與比較

報告：[7. 混淆矩陣](docs/Report.md#7-模型評估混淆矩陣)、[8. PCA vs LDA](docs/Report.md#8-降維比較pca-vs-lda)、[9. 決策邊界比較](docs/Report.md#9-決策邊界比較)

| 程式 | 功能 | 輸出 |
|---|---|---|
| `01_confusion_matrices.py` | 畫出 KNN (k=17)、LDA (皆用 10-fold 交叉驗證預測) 與 K-Means (k=3) 的混淆矩陣，並印出各類別的 precision / recall | 文字 + 1 張圖 |
| `02_pca_vs_lda.py` | 比較 PCA (非監督式) 與 LDA (監督式) 的二維投影、第一成分直方圖、解釋變異比例，並以 KNN 交叉驗證比較投影後的準確率 | 文字 + 2 張圖 |
| `03_decision_boundaries.py` | 以花瓣長度、寬度畫出 KNN (k=1、k=17)、LDA、K-Means 的決策邊界並標出誤判樣本；圖與輸出皆為中文 | 文字 + 1 張圖 |

## 環境建置

以 Python 3.10 測試，建立虛擬環境並安裝相同版本的套件：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 執行

所有路徑皆以專案根目錄為基準，可在任何目錄下執行：

```bash
python 01_data_exploration/01_dataset_info.py
```

有畫圖的程式會把每張圖存成 `outputs/<檔名>_<編號>.png`；有圖形介面時也會另外跳出視窗。

`03_decision_boundaries.py` 的圖使用中文字型，會依序尋找 Noto Sans CJK TC、Microsoft JhengHei (Windows)、PingFang TC (macOS)。Linux 若沒有中文字型，可安裝 `fonts-noto-cjk`。

一次執行全部程式：

```bash
for f in 0*/*.py; do echo "== $f"; python "$f"; done
```
