import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置
# ============================================================
DATA_PATH = 'D:\\Rstudio\\Rmyfile\\Traditional Chinese Medicine\\ec50-12-26.xlsx'
SHEET = 3
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_BOOTSTRAP = 1000  # bootstrap 迭代次数
THRESHOLDS = [100, 300]  # 敏感性分析的阈值

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
print(f"数据集大小: {df.shape[0]} 化合物, {df.shape[1]} 列")
print(f"EC50 范围: {df['ec50'].min():.2f} - {df['ec50'].max():.2f} μM")

# ============================================================
# 2. 多阈值敏感性分析
# 存储完整对象：最优模型、TestX、TestY、预测概率、bootstrap auc数组，用于后续绘图导出，消除断层
# ============================================================
print("\n" + "=" * 60)
print("2. 多阈值敏感性分析")
print("=" * 60)

# 每个阈值保存完整全套对象，不再只存指标
results_summary = {}

for threshold in THRESHOLDS:
    print(f"\n--- 阈值 = {threshold} μM ---")

    # 准备数据
    Y = (df['ec50'] <= threshold).astype(int)
    X = df.drop(columns=['ec50'])

    n_active = Y.sum()
    n_inactive = len(Y) - n_active
    print(f"  活性化合物: {n_active}, 非活性: {n_inactive} (比例: {n_active/len(Y):.1%})")

    # Train/test split (先拆分!)
    trainx, TestX, trainy, TestY = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )

    # SMOTE 仅在训练集上
    sm = SMOTE(random_state=RANDOM_STATE)
    TrainX, TrainY = sm.fit_resample(trainx, trainy)
    print(f"  SMOTE后训练集: {Counter(TrainY)}")

    # 训练5个模型 GridSearchCV
    model_dict = {}

    # RF
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(TrainX, TrainY)
    model_dict['RF'] = grid_rf.best_estimator_

    # XGB
    xgb = XGBClassifier(random_state=RANDOM_STATE, verbosity=0)
    param_grid_xgb = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8]
    }
    grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_xgb.fit(TrainX, TrainY)
    model_dict['XGB'] = grid_xgb.best_estimator_

    # GBM
    gbm = GradientBoostingClassifier(random_state=RANDOM_STATE)
    param_grid_gbm = {
        'n_estimators': [100],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 4],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_gbm = GridSearchCV(gbm, param_grid_gbm, cv=10, scoring='accuracy', n_jobs=-1)
    grid_gbm.fit(TrainX, TrainY)
    model_dict['GBM'] = grid_gbm.best_estimator_

    # AdaBoost
    ada = AdaBoostClassifier(random_state=RANDOM_STATE)
    param_grid_ada = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.5, 1]
    }
    grid_ada = GridSearchCV(ada, param_grid_ada, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_ada.fit(TrainX, TrainY)
    model_dict['AdaBoost'] = grid_ada.best_estimator_

    # ANN
    ann = MLPClassifier(random_state=1, max_iter=300)
    param_grid_ann = {
        'hidden_layer_sizes': [(50,), (100,), (200,)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],
        'batch_size': [16, 32, 64],
        'learning_rate': ['constant', 'invscaling'],
    }
    grid_ann = GridSearchCV(ann, param_grid_ann, cv=10, scoring='accuracy', n_jobs=-1)
    grid_ann.fit(TrainX, TrainY)
    model_dict['ANN'] = grid_ann.best_estimator_

    # 评估每个模型 + Bootstrap AUC 95% CI【百分位数bootstrap，抛弃正态近似】
    threshold_results = {}
    for name, model in model_dict.items():
        y_prob = model.predict_proba(TestX)[:, 1]
        y_pred = model.predict(TestX)

        # 基础指标
        fpr, tpr, _ = roc_curve(TestY, y_prob)
        auc_val = auc(fpr, tpr)
        acc = accuracy_score(TestY, y_pred)

        # ----------------------
        # Bootstrap AUC 百分位数CI，补采样保证达到N_BOOTSTRAP次有效采样
        # ----------------------
        n_test = len(TestY)
        aucs_bootstrap = []
        rng = np.random.RandomState(RANDOM_STATE)
        while len(aucs_bootstrap) < N_BOOTSTRAP:
            idx = rng.randint(0, n_test, n_test)
            y_true_boot = TestY.iloc[idx]
            y_prob_boot = y_prob[idx]
            if len(np.unique(y_true_boot)) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_true_boot, y_prob_boot)
            aucs_bootstrap.append(auc(fpr_b, tpr_b))

        aucs_bootstrap = np.array(aucs_bootstrap)
        # percentile bootstrap 95%CI，不再正态近似
        auc_ci_low = np.percentile(aucs_bootstrap, 2.5)
        auc_ci_high = np.percentile(aucs_bootstrap, 97.5)

        threshold_results[name] = {
            'model': model,
            'TestX': TestX.copy(),
            'TestY': TestY.copy(),
            'y_prob': y_prob.copy(),
            'AUC': auc_val,
            'AUC_95CI_low': auc_ci_low,
            'AUC_95CI_high': auc_ci_high,
            'auc_boot_array': aucs_bootstrap.copy(),
            'Accuracy': acc,
        }

        print(f"  {name:10s}: AUC={auc_val:.3f} (95%CI: {auc_ci_low:.3f}-{auc_ci_high:.3f}), Acc={acc:.3f}")

    results_summary[threshold] = threshold_results

# ============================================================
# 3. 敏感性分析汇总打印
# ============================================================
print("\n" + "=" * 60)
print("3. 敏感性分析汇总: 不同阈值下的AUC变化")
print("=" * 60)

print(f"\n{'Model':<10s}", end="")
for t in THRESHOLDS:
    print(f"  {'EC50≤'+str(t)+'μM':>20s}", end="")
print()

for name in ['RF', 'XGB', 'GBM', 'AdaBoost', 'ANN']:
    print(f"{name:<10s}", end="")
    for t in THRESHOLDS:
        auc_val = results_summary[t][name]['AUC']
        ci_low = results_summary[t][name]['AUC_95CI_low']
        ci_high = results_summary[t][name]['AUC_95CI_high']
        print(f"  {auc_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]", end="")
    print()

# ============================================================
# 4. Figure3B ROC绘图：直接复用300μM已经训练好的全套结果，不再重新划分、重新训练
# ROC置信带：TPR使用bootstrap百分位数，不用mean±1.96*std正态近似
# ============================================================
print("\n" + "=" * 60)
print("4. 绘制ROC曲线 (300 μM 阈值, Bootstrap 95%CI)")
print("=" * 60)

threshold_main = 300
res_300 = results_summary[threshold_main]
colors = {'RF': '#2196F3', 'XGB': '#F44336', 'GBM': '#FF9800', 'AdaBoost': '#9C27B0', 'ANN': '#4CAF50'}

plt.figure(figsize=(8, 6))
mean_fpr = np.linspace(0, 1, 100)

for name, color in colors.items():
    item = res_300[name]
    TestY = item['TestY']
    y_prob = item['y_prob']
    fpr, tpr, _ = roc_curve(TestY, y_prob)

    # ROC bootstrap，插值TPR，百分位数获取置信边界
    n_test = len(TestY)
    tprs_bootstrap = []
    rng = np.random.RandomState(RANDOM_STATE)
    while len(tprs_bootstrap) < N_BOOTSTRAP:
        idx = rng.randint(0, n_test, n_test)
        y_true_boot = TestY.iloc[idx]
        y_prob_boot = y_prob[idx]
        if len(np.unique(y_true_boot)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true_boot, y_prob_boot)
        interp_tpr = np.interp(mean_fpr, fpr_b, tpr_b)
        interp_tpr[0] = 0.0
        tprs_bootstrap.append(interp_tpr)

    tprs_bootstrap = np.array(tprs_bootstrap)
    tpr_lower = np.percentile(tprs_bootstrap, 2.5, axis=0)
    tpr_upper = np.percentile(tprs_bootstrap, 97.5, axis=0)
    tpr_lower[tpr_lower < 0] = 0.0
    tpr_upper[tpr_upper > 1] = 1.0

    auc_val = item['AUC']
    auc_low = item['AUC_95CI_low']
    auc_high = item['AUC_95CI_high']

    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC={auc_val:.2f}, 95%CI [{auc_low:.2f}, {auc_high:.2f}])')
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.15)

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves with Bootstrap 95% Confidence Intervals (EC₅₀ ≤ 300 μM)', fontsize=13)
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig('Figure_ROC_with_95CI.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Figure_ROC_with_95CI.png', dpi=300, bbox_inches='tight')
print("已保存: Figure_ROC_with_95CI.pdf/png")
plt.close()

# ============================================================
# 5. 导出Supplementary Table，完全复用循环内计算结果，无二次计算
# ============================================================
print("\n" + "=" * 60)
print("5. 导出 Supplementary Table: 多阈值模型性能对比")
print("=" * 60)

table_rows = []
for t in THRESHOLDS:
    Y_t = (df['ec50'] <= t).astype(int)
    n_active = Y_t.sum()
    n_total = len(Y_t)
    active_ratio = n_active / n_total
    for name in ['RF', 'XGB', 'GBM', 'AdaBoost', 'ANN']:
        r = results_summary[t][name]
        table_rows.append({
            'Threshold (μM)': t,
            'N_active': n_active,
            'N_total': n_total,
            'Active_ratio': f"{active_ratio:.1%}",
            'Model': name,
            'AUC': round(r['AUC'], 3),
            'AUC_95CI_low': round(r['AUC_95CI_low'], 3),
            'AUC_95CI_high': round(r['AUC_95CI_high'], 3),
            'Accuracy': round(r['Accuracy'], 3),
        })

table_df = pd.DataFrame(table_rows)
table_df.to_csv('Supplementary_Table_Threshold_Sensitivity.csv', index=False)
print(table_df.to_string(index=False))
print("\n已保存: Supplementary_Table_Threshold_Sensitivity.csv")

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)
