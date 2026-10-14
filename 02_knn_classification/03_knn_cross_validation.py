# Choose k for KNN with repeated stratified 10-fold cross-validation, comparing training and validation accuracy
# 以重複分層 10-fold 交叉驗證選擇 KNN 的 k 值，並比較訓練與驗證準確率

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)

# Candidate k values
# 候選的 k 值
K_RANGE = range(1, 31)

# 10 folds x 10 repeats = 100 train/validation splits per k; stratified keeps 1/3 of each species in every fold
# 每個 k 做 10 折 x 重複 10 次 = 100 次訓練/驗證；分層抽樣讓每一折中三個品種的比例相同
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)

train_mean, test_mean, test_std = [], [], []
for k in K_RANGE:
    result = cross_validate(KNeighborsClassifier(n_neighbors=k), X, y, cv=cv, return_train_score=True)
    train_mean.append(result['train_score'].mean())
    test_mean.append(result['test_score'].mean())
    test_std.append(result['test_score'].std())
    print(f'k={k:2d}  train={train_mean[-1]:.3f}  validation={test_mean[-1]:.3f} ± {test_std[-1]:.3f}')

test_mean = np.array(test_mean)
test_std = np.array(test_std)
best_k = K_RANGE[int(test_mean.argmax())]
print(f'Best k / 最佳 k: {best_k}  validation accuracy / 驗證準確率 = {test_mean.max():.3f} ± {test_std[test_mean.argmax()]:.3f}')

plt.figure(figsize=(9, 5))
plt.plot(K_RANGE, train_mean, 's--', color='gray', label='Training accuracy')
plt.plot(K_RANGE, test_mean, 'o-', color='tab:blue', label='Validation accuracy (mean)')
# Shaded band: validation accuracy ± 1 standard deviation across the 100 splits
# 陰影帶：100 次切分的驗證準確率 ± 1 個標準差
plt.fill_between(K_RANGE, test_mean - test_std, test_mean + test_std, color='tab:blue', alpha=0.15,
                 label='Validation ± 1 std')
plt.axvline(best_k, color='tab:red', linestyle=':', label=f'Best k = {best_k}')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN accuracy vs. k (10-fold x 10 repeated CV)')
plt.legend(loc='lower left')
# What each saved figure shows, printed together with its path
# 每張圖的意義，會與存檔路徑一起印出
FIGURE_DESCRIPTIONS = {
    1: (f'KNN accuracy for k = 1-30: gray = training accuracy, blue = validation accuracy (mean ± 1 std over 100 CV splits). k = 1 overfits (100% training, lower validation); large k underfits (both drop). Best k = {best_k}.',
        f'k = 1-30 的 KNN 準確率：灰 = 訓練準確率，藍 = 驗證準確率 (100 次交叉驗證的平均 ± 1 標準差)。k = 1 過擬合 (訓練 100%、驗證較低)；k 太大則欠擬合 (兩者一起下降)。最佳 k = {best_k}。'),
}
# Save every figure to the project's outputs/ folder (keeps results even without a display)
# 將所有圖存到專案的 outputs/ 資料夾 (無圖形介面時也能保留結果)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
for num in plt.get_fignums():
    out_path = OUTPUT_DIR / f'{Path(__file__).stem}_{num}.png'
    plt.figure(num).savefig(out_path, bbox_inches='tight')
    print('Saved figure / 已儲存圖檔:', out_path)
    en, zh = FIGURE_DESCRIPTIONS[num]
    print('  ' + en)
    print('  ' + zh)
plt.show()
