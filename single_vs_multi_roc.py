"""
单靶点 vs 多靶点（ROC 对比）
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

RS = 42
TEST_SIZE = 0.25
THRESHOLD = 300
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=RS)

# ============================================================
# 1. 加载数据（240 全集）
# ============================================================
df = pd.read_excel('/Users/sam/D/共同研究者/JTM/ec50-12-26.xlsx', sheet_name=3)
target_cols = [c for c in df.columns if c not in ['ec50']]
Y = (df['ec50'] <= THRESHOLD).astype(int)
X_multi = df[target_cols]                       # 20 靶点
X_single = X_multi.max(axis=1).to_frame()       # 单靶点最优分数（1 特征）
print(f"全集化合物: {len(df)}, 靶点数: {len(target_cols)}")

# ============================================================
# 2. 相同的随机拆分（保证测试集一致）
# ============================================================
tr_idx, te_idx = train_test_split(
    np.arange(len(Y)), test_size=TEST_SIZE, random_state=RS, stratify=Y)

sm = SMOTE(random_state=RS)

def fit_predict(X_tr, Y_tr, X_te):
    Xb, Yb = sm.fit_resample(X_tr, Y_tr)
    grid = GridSearchCV(RandomForestClassifier(random_state=RS), PARAM_GRID,
                        cv=CV, scoring='accuracy', n_jobs=-1)
    grid.fit(Xb, Yb)
    return grid.best_estimator_.predict_proba(X_te)[:, 1]

proba_single = fit_predict(X_single.iloc[tr_idx], Y.iloc[tr_idx], X_single.iloc[te_idx])
proba_multi = fit_predict(X_multi.iloc[tr_idx], Y.iloc[tr_idx], X_multi.iloc[te_idx])

# ============================================================
# 3. ROC + AUC
# ============================================================
fpr_s, tpr_s, _ = roc_curve(Y.iloc[te_idx], proba_single)
auc_s = auc(fpr_s, tpr_s)
acc_s = accuracy_score(Y.iloc[te_idx], (proba_single >= 0.5).astype(int))

fpr_m, tpr_m, _ = roc_curve(Y.iloc[te_idx], proba_multi)
auc_m = auc(fpr_m, tpr_m)
acc_m = accuracy_score(Y.iloc[te_idx], (proba_multi >= 0.5).astype(int))

print(f"Single-best-target : AUC = {auc_s:.3f}, Accuracy = {acc_s:.3f}")
print(f"Multi-target       : AUC = {auc_m:.3f}, Accuracy = {acc_m:.3f}")
print(f"ΔAUC (multi - single) = {auc_m - auc_s:+.3f}")

# ============================================================
# 4. 绘图
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(7.5, 6.5))

ax.plot(fpr_m, tpr_m, color='#1E3A5F', lw=2.4,
        label='Multi-target (20 targets)\nAUC = {:.3f}'.format(auc_m))
ax.plot(fpr_s, tpr_s, color='#C62828', lw=2.2, linestyle='--',
        label='Single-best-target (1 target)\nAUC = {:.3f}'.format(auc_s))
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.55)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Multi-target vs Single-best-target', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/Users/sam/D/共同研究者/JTM/Figure_Single_vs_Multi_ROC.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/sam/D/共同研究者/JTM/Figure_Single_vs_Multi_ROC.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存: Figure_Single_vs_Multi_ROC.pdf / .png")
