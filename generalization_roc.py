"""
泛化能力 — Random split vs Scaffold split（ROC 对比）
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
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
# 1. 加载数据 + 骨架
# ============================================================
df = pd.read_excel('/Users/sam/D/共同研究者/JTM/ec50-12-26.xlsx', sheet_name=3)
target_cols = [c for c in df.columns if c not in ['ec50']]
smiles_df = pd.read_csv('/Users/sam/D/共同研究者/JTM/training_set_smiles.csv')

valid_idx = []
scaffolds = []
for i, s in enumerate(smiles_df['SMILES']):
    if isinstance(s, str) and len(s) > 0:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaff) if scaff else 'NONE')
            valid_idx.append(i)

X_all = df[target_cols].iloc[valid_idx].values
Y_all = (df['ec50'].iloc[valid_idx] <= THRESHOLD).astype(int).values
print(f"能算骨架的化合物: {len(valid_idx)} / {len(df)}")

sm = SMOTE(random_state=RS)

def fit_predict(X_tr, Y_tr, X_te):
    Xb, Yb = sm.fit_resample(X_tr, Y_tr)
    grid = GridSearchCV(RandomForestClassifier(random_state=RS), PARAM_GRID,
                        cv=CV, scoring='accuracy', n_jobs=-1)
    grid.fit(Xb, Yb)
    return grid.best_estimator_.predict_proba(X_te)[:, 1]

# ============================================================
# 2. Random split vs Scaffold split
# ============================================================
tr_idx, te_idx = train_test_split(
    np.arange(len(Y_all)), test_size=TEST_SIZE, random_state=RS, stratify=Y_all)
proba_random = fit_predict(X_all[tr_idx], Y_all[tr_idx], X_all[te_idx])

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RS)
tr_sc, te_sc = next(gss.split(X_all, Y_all, groups=scaffolds))
proba_scaffold = fit_predict(X_all[tr_sc], Y_all[tr_sc], X_all[te_sc])

# ============================================================
# 3. ROC + AUC
# ============================================================
fpr_r, tpr_r, _ = roc_curve(Y_all[te_idx], proba_random)
auc_r = auc(fpr_r, tpr_r)
acc_r = accuracy_score(Y_all[te_idx], (proba_random >= 0.5).astype(int))

fpr_s, tpr_s, _ = roc_curve(Y_all[te_sc], proba_scaffold)
auc_s = auc(fpr_s, tpr_s)
acc_s = accuracy_score(Y_all[te_sc], (proba_scaffold >= 0.5).astype(int))

print(f"Random split   : AUC = {auc_r:.3f}, Accuracy = {acc_r:.3f}")
print(f"Scaffold split : AUC = {auc_s:.3f}, Accuracy = {acc_s:.3f}")
print(f"ΔAUC (scaffold - random) = {auc_s - auc_r:+.3f}")

# ============================================================
# 4. 绘图
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(7.5, 6.5))

ax.plot(fpr_r, tpr_r, color='#1E3A5F', lw=2.4,
        label='Random split\nAUC = {:.3f}'.format(auc_r))
ax.plot(fpr_s, tpr_s, color='#00796B', lw=2.2, linestyle='-.',
        label='Scaffold split\nAUC = {:.3f}'.format(auc_s))
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.55)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Generalization: Random split vs Scaffold split', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/Users/sam/D/共同研究者/JTM/Figure_Generalization_ROC.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/sam/D/共同研究者/JTM/Figure_Generalization_ROC.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存: Figure_Generalization_ROC.pdf / .png")
