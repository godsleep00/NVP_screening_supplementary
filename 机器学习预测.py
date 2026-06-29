# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier  
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# %% [markdown]
# # 数据预处理

# %%
EC50 = pd.read_excel("D:\\Rstudio\\Rmyfile\\Traditional Chinese Medicine\\ec50-12-26.xlsx", sheet_name=3)
print(f"Number of rows: {EC50.shape[0]}, Number of columns: {EC50.shape[1]}")
EC50['result'] = EC50['ec50'].apply(lambda x: 1 if x <= 300 else 0)   # x《300记为1
print(EC50)

# %%
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为SimHei，支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
sns.countplot(x='result', data=EC50)
plt.show()
print('Resampled dataset shape %s' % Counter(EC50["result"]))

# %%
X = EC50.drop(columns=['result','ec50'])  # 'result'是目标变量的新名称，'ec50'用于生成'result'
Y = EC50['result']

# %% [markdown]
# ## 划分训练集和测试集

# %%
np.random.seed(1234)
trainx, TestX, trainy, TestY = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)
print('训练集和测试集划分完毕==================================5')
trainx

# %% [markdown]
# ## 过采样SMOTE

# %%
from imblearn.over_sampling import SMOTE
from collections import Counter

# %%
#from imblearn.over_sampling import ADASYN  # 导入ADASYN

# 创建ADASYN实例
#adasyn = ADASYN(random_state=42)

# 进行过采样
#TrainX, TrainY = adasyn.fit_resample(trainx, trainy)

# 查看过采样后数据集的类别分布
#print('Resampled dataset shape %s' % Counter(TrainY))

# %%
sm = SMOTE(random_state=42)

# 进行过采样
TrainX, TrainY = sm.fit_resample(trainx,trainy )

# 查看过采样后数据集的类别分布
print('Resampled dataset shape %s' % Counter(TrainY))

# %%
df = TrainY.to_frame(name='result')

# 使用转换后的 DataFrame 绘制计数图
sns.countplot(x='result', data=df)
plt.show()

# %% [markdown]
# # 机器学习

# %%
# 创建重复三次的5折交叉验证器
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
def myconfusionmatrix(TestY, y_pred):
    # 混淆矩阵
    cm = confusion_matrix(TestY, y_pred)
    # Kappa值
    kappa = cohen_kappa_score(TestY, y_pred)
    # 计算准确率
    accuracy = accuracy_score(TestY, y_pred)
    # 计算精确率（对于多分类问题，需要指定平均方法，如'macro', 'micro', 'weighted'）
    precision = precision_score(TestY, y_pred, average='macro')  # 或者使用'micro', 'weighted'等
    # 计算召回率
    recall = recall_score(TestY, y_pred, average='macro')  # 同样需要指定平均方法
    # 计算F1分数
    f1 = f1_score(TestY, y_pred, average='weighted')  # 同样需要指定平均方法
    return (cm, accuracy, precision, recall, f1, kappa)

# %% [markdown]
# ## 随机森林RF

# %%

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}
rf_classifier = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}
rf_classifier = RandomForestClassifier(random_state=42)

# %%
grid_search = GridSearchCV(rf_classifier, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(TrainX, TrainY)

# %%
print("Best parameters:", grid_search.best_params_)

# 使用最优参数创建新的随机森林分类器
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rf.fit(TrainX, TrainY)

# %%
y_pred = best_rf.predict(TestX)
list2 = myconfusionmatrix(TestY, y_pred)
confusion_matrix = list2[0]
accuracy = list2[1]
precision = list2[2]
recall = list2[3]
f1_score = list2[4]
Kappa = list2[5]
print('输出混淆矩阵========================================7')
print(f"Confusion Matrix:\n{confusion_matrix}\nKappa:{Kappa}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}")

# %% [markdown]
# ## XGBoost

# %%
from xgboost import XGBClassifier  
from sklearn.pipeline import Pipeline  
from sklearn.model_selection import GridSearchCV, StratifiedKFold  

# %%
# 定义XGBoost模型  
xgb_model = XGBClassifier() 

# %%
# 定义参数网格进行网格搜索    
param_grid4 = {
    # 树的数量，增加更多不同的取值，以探索不同数量的树对模型性能的影响
    'n_estimators': [100, 150, 200],
    # 树的最大深度，扩大深度范围，观察不同深度对模型复杂度和性能的影响
    'max_depth': [ 4, 5, 6],
    # 学习率，增加更小和稍大的学习率取值，学习率影响模型收敛速度和精度
    'learning_rate': [ 0.01, 0.05, 0.1],
    # 子样本比例，增加更多取值，用于控制训练每棵树时使用的样本比例，防止过拟合
    'subsample': [ 0.7, 0.8, 0.9],
    # 每棵树使用的特征比例，提供更多选择，帮助选择合适的特征比例构建树
    'colsample_bytree': [ 0.6, 0.7, 0.8]
}
# 使用GridSearchCV进行交叉验证和参数调优  
grid_xgb = GridSearchCV(xgb_model, param_grid4, cv=cv, scoring='accuracy', n_jobs=-1)  
grid_xgb.fit(TrainX, TrainY)  
  
# 输出最佳参数和最佳得分  
print("Best parameters found: ", grid_xgb.best_params_)  
print("Best cross-validation score: {:.2f}".format(grid_xgb.best_score_))
# 选择最优参数
best_xgb = XGBClassifier(**grid_xgb.best_params_, random_state=42)
best_xgb.fit(TrainX, TrainY)

# %%
# 创建SHAP解释器
import shap
import matplotlib.pyplot as plt
plt.clf()
explainer = shap.TreeExplainer(best_xgb)

# 计算SHAP值
shap_values = explainer.shap_values(TrainX)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 10  # 设置字体大小为14
# 现在创建 SHAP 可视化 

#配色   viridis  Spectral   coolwarm  RdYlGn  RdYlBu  RdBu  RdGy  PuOr  BrBG PRGn  PiYG 
shap.summary_plot(shap_values, TrainX,show=False)

plt.savefig('importance5.pdf')
#粉红色点：表示该特征值在这个观察中对模型预测产生了正面影响（增加预测值）
#蓝色点：表示该特征值在这个观察中对模型预测产生了负面影响（降低预测值）
#水平轴（SHAP 值）显示了影响的大小。点越远离中心线（零点），该特征对模型输出的影响越大
#图中垂直排列的特征按影响力从上到下排序。上方的特征对模型输出的总体影响更大，而下方的特征影响较小。
# 最上方的特征（例如 "lstat"）显示了大量的正面和负面影响，表明它在不同的观察值中对模型预测的结果有很大的不同影响。
# 中部的特征（如 "rm" 和 "dis"）也显示出两种颜色的点，但点的分布更集中，影响相对较小。
# 底部的特征（如 "chas" 和 "zn"）对模型的影响最小，且大部分影响较为接近零，表示这些特征对模型预测的贡献较小
plt.show()

# %%
plt.clf()
shap.summary_plot(shap_values, TrainX,plot_type='bar',show=False)
#主要表示绝对重要值的大小，把SHAP value 的样本取了绝对平均值
# 房价预测模型中，lstat（较低状态人口的百分比）和 rm（住宅平均房间数）可能是影响房价的关键因素
plt.savefig('importance3.pdf')

# %%
shap_values2 = explainer(TrainX)
plt.clf()
# 设置字体和颜色
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 12,  # 改变字体大小
    'text.color': 'black',  # 修改文字颜色
    'axes.labelcolor': 'green',  # 修改轴标签颜色
    'xtick.color': 'red',  # 修改x轴刻度颜色
    'ytick.color': 'blue'  # 修改y轴刻度颜色
})

# 确保 shap_values2 是 DataFrame 格式，并已正确计算
shap.plots.heatmap(shap_values2,show=False)

# 显示图表

plt.savefig('importance4.pdf', dpi=300, bbox_inches='tight')  # 高分辨率保存

plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier  
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# %%
y_pred4 = best_xgb.predict(TestX)
list4 = myconfusionmatrix(TestY, y_pred4)
confusion_matrix4 = list4[0]
accuracy4 = list4[1]
precision4 = list4[2]
recall4 = list4[3]
f1_score4 = list4[4]
Kappa4 = list4[5]
print('输出混淆矩阵========================================7')
print(f"Confusion Matrix:\n{confusion_matrix4}\nKappa:{Kappa4}\nAccuracy: {accuracy4}\nPrecision: {precision4}\nRecall: {recall4}\nF1 Score: {f1_score4}")

# %% [markdown]
# ## 梯度提升机GBM

# %%
# 定义梯度提升机模型  
gbm = GradientBoostingClassifier(random_state=42)  
  
# 定义参数网格进行网格搜索  
param_grid3 = {  
    'n_estimators': [100],  
    'learning_rate': [0.1,0.05],  
    'max_depth': [3, 4],  
    'min_samples_split': [2, 5],  
    'min_samples_leaf': [1, 2]  
}  
  
# 使用GridSearchCV进行网格搜索和交叉验证  
grid_gbm = GridSearchCV(gbm, param_grid=param_grid3, cv=10, scoring='accuracy', n_jobs=-1)  
  
  
# 训练模型  
grid_gbm.fit(TrainX, TrainY)  
  
# 输出最佳参数  
print("Best parameters found: ", grid_gbm.best_params_)  

# 使用最佳模型进行预测  
best_gbm = grid_gbm.best_estimator_  
y_pred = best_gbm.predict(TestX)  

# %%
y_pred3 = best_gbm.predict(TestX)
list3 = myconfusionmatrix(TestY, y_pred3)
confusion_matrix3 = list3[0]
accuracy3 = list3[1]
precision3 = list3[2]
recall3 = list3[3]
f1_score3 = list3[4]
Kappa3 = list3[5]
print('输出混淆矩阵========================================7')
print(f"Confusion Matrix:\n{confusion_matrix3}\nKappa:{Kappa3}\nAccuracy: {accuracy3}\nPrecision: {precision3}\nRecall: {recall3}\nF1 Score: {f1_score3}")

# %% [markdown]
# ## Adaboost

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer

# %%
param_grid5 = {
       'n_estimators': [50, 100, 150],
       'learning_rate': [0.1, 0.5, 1]
}
ada_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)

# %%

# 使用GridSearchCV进行交叉验证和参数调优  
grid_ada = GridSearchCV(ada_classifier, param_grid5, cv=cv, scoring='accuracy', n_jobs=-1)  
grid_ada.fit(TrainX, TrainY)  
  
# 输出最佳参数和最佳得分  
print("Best parameters found: ", grid_ada.best_params_)  
print("Best cross-validation score: {:.2f}".format(grid_ada.best_score_))
# 选择最优参数
best_ada = grid_ada.best_estimator_ 

# %%
y_pred5 = best_ada.predict(TestX)
list5 = myconfusionmatrix(TestY, y_pred5)
confusion_matrix5 = list5[0]
accuracy5 = list5[1]
precision5 = list5[2]
recall5 = list5[3]
f1_score5 = list5[4]
Kappa5= list5[5]
print('输出混淆矩阵========================================7')
print(f"Confusion Matrix:\n{confusion_matrix5}\nKappa:{Kappa5}\nAccuracy: {accuracy5}\nPrecision: {precision5}\nRecall: {recall5}\nF1 Score: {f1_score5}")

# %% [markdown]
# ## 人工神经网络ANN

# %%
from sklearn.neural_network import MLPClassifier  

# %%
# 定义ANN模型  
ann_model = MLPClassifier(random_state=1, max_iter=300)  # 设置随机种子和最大迭代次数  

  
# 定义参数网格进行网格搜索  

param_grid2 = {  
    'hidden_layer_sizes': [(50,), (100,),(200,)],  # 减少隐藏层配置的复杂度，先尝试单层结构  
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    # L2惩罚参数：尝试多个不同的值
    'alpha': [0.0001, 0.001],
    # 批量大小：尝试更多不同的批量大小
    'batch_size': [16, 32, 64],
    # 学习率策略：除了constant，还增加了invscaling
    'learning_rate': ['constant', 'invscaling'],
}


# %%
 # 设置 GridSearchCV  
grid_ann = GridSearchCV(estimator=ann_model, param_grid=param_grid2, scoring='accuracy', cv=10, n_jobs=-1)  
  
# 拟合模型
grid_ann.fit(TrainX, TrainY) 

best_ann = grid_ann.best_estimator_ 
print("Best parameters for ANN:", grid_ann.best_params_)
print("Best cross-validation score for ANN:", grid_ann.best_score_)

# %%
y_pred1 = best_ann.predict(TestX)
list2 = myconfusionmatrix(TestY, y_pred1)
confusion_matrix1 = list2[0]
accuracy1 = list2[1]
precision1 = list2[2]
recall1 = list2[3]
f1_score1 = list2[4]
Kappa1 = list2[5]
print('输出混淆矩阵========================================7')
print(f"Confusion Matrix:\n{confusion_matrix1}\nKappa:{Kappa1}\nAccuracy: {accuracy1}\nPrecision: {precision1}\nRecall: {recall1}\nF1 Score: {f1_score1}")

# %% [markdown]
# # ROC曲线绘制

# %% [markdown]
# ## 测试集上的表现

# %%
from sklearn.metrics import roc_curve, auc

# %%
rf_pred_prob = best_rf.predict_proba(TestX)[:, 1] 
ann_pred_prob = best_ann.predict_proba(TestX)[:, 1] 
gbm_pred_prob = best_gbm.predict_proba(TestX)[:, 1] 
xgb_pred_prob = best_xgb.predict_proba(TestX)[:, 1] 
ada_pred_prob = best_ada.predict_proba(TestX)[:, 1] 

# %%
fpr1, tpr1, thresholds1 = roc_curve(TestY, rf_pred_prob)  
fpr2, tpr2, thresholds2 = roc_curve(TestY, ann_pred_prob)  
fpr3, tpr3, thresholds3 = roc_curve(TestY, gbm_pred_prob)  
fpr4, tpr4, thresholds4 = roc_curve(TestY, xgb_pred_prob)
fpr5, tpr5, thresholds5 = roc_curve(TestY, ada_pred_prob)
auc1 = auc(fpr1, tpr1)  
auc2 = auc(fpr2, tpr2) 
auc3 = auc(fpr3, tpr3)  
auc4 = auc(fpr4, tpr4) 
auc5 = auc(fpr5, tpr5) 

# %%
plt.figure()  
plt.plot(fpr1, tpr1, color='blue', lw=1, label=f'Random Forest (AUC = {auc1:.2f})')  
plt.plot(fpr2, tpr2, color='green', lw=1, label=f'Artificial Neural Network (AUC = {auc2:.2f})') 
plt.plot(fpr5, tpr5, color='purple', lw=1, label=f'Adaptive Boosting (AUC = {auc5:.2f})')  
plt.plot(fpr3, tpr3, color='orange', lw=1, label=f'Gradient Boosting Machine (AUC = {auc3:.2f})')  
plt.plot(fpr4, tpr4, color='red', lw=1, label=f'Extreme Gradient Boosting (AUC = {auc4:.2f})') 

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver Operating Characteristic (ROC) Curves')  
plt.legend(loc="lower right") 
plt.savefig('roc_curves.pdf', dpi=300, bbox_inches='tight') 
plt.show()


# %% [markdown]
# # 堆叠模型

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# --------------------------- 1. 提取基础模型预测概率（元特征准备） ---------------------------
# 假设已存在以下对象：
# 基础模型：best_rf, best_xgb, best_gbm, best_ada, best_ann
# 数据集：TrainX(训练特征), TrainY(训练标签), TestX(测试特征), TestY(测试标签), new_data(新数据)

def get_model_probs(model, data):
    """获取模型对数据的预测概率（正类）"""
    # 根据不同模型类型获取预测概率
    if hasattr(model, 'predict_proba'):
        # 大多数分类器的标准接口
        probs = model.predict_proba(data)[:, 1]
    elif hasattr(model, 'predict') and 'xgb' in str(type(model)).lower():
        # XGBoost的特殊处理
        probs = model.predict(data)
        # 如果是原始输出分数，转换为概率
        if probs.min() < 0 or probs.max() > 1:
            probs = 1 / (1 + np.exp(-probs))
    else:
        raise ValueError(f"不支持的模型类型: {type(model)}")
    return probs

# 2.1 为训练集提取各模型预测概率（用于训练元分类器）
rf_probs_train = get_model_probs(best_rf, TrainX)
xgb_probs_train = get_model_probs(best_xgb, TrainX)
gbm_probs_train = get_model_probs(best_gbm, TrainX)
ada_probs_train = get_model_probs(best_ada, TrainX)
ann_probs_train = get_model_probs(best_ann, TrainX)

# 组合成元特征矩阵（训练集）
meta_train = pd.DataFrame({
    'rf_prob': rf_probs_train,
    'xgb_prob': xgb_probs_train,
    'gbm_prob': gbm_probs_train,
    'ada_prob': ada_probs_train,
    'ann_prob': ann_probs_train
})

# 2.2 为测试集提取各模型预测概率（用于验证堆叠效果）
rf_probs_test = get_model_probs(best_rf, TestX)
xgb_probs_test = get_model_probs(best_xgb, TestX)
gbm_probs_test = get_model_probs(best_gbm, TestX)
ada_probs_test = get_model_probs(best_ada, TestX)
ann_probs_test = get_model_probs(best_ann, TestX)

# 组合成元特征矩阵（测试集）
meta_test = pd.DataFrame({
    'rf_prob': rf_probs_test,
    'xgb_prob': xgb_probs_test,
    'gbm_prob': gbm_probs_test,
    'ada_prob': ada_probs_test,
    'ann_prob': ann_probs_test
})


# --------------------------- 2. 训练元分类器（逻辑回归） ---------------------------
# 使用带交叉验证的逻辑回归，自动选择最优正则化参数
lr_meta = LogisticRegressionCV(
    Cs=10,  # 正则化强度参数的候选值数量
    cv=10,  # 交叉验证折数
    penalty='l2',  # L2正则化（Ridge），可改为'l1'使用Lasso
    solver='liblinear',  # 小型数据集适用的求解器
    scoring='accuracy',  # 评估指标
    random_state=42,
    max_iter=1000
)

# 训练元分类器
lr_meta.fit(meta_train, TrainY)

# 输出最优正则化参数
print(f"最优正则化参数: {lr_meta.C_[0]}")

# --------------------------- 3. 模型评估与预测 ---------------------------
# 3.1 测试集评估（验证堆叠效果）
test_pred_prob = lr_meta.predict_proba(meta_test)[:, 1]  # 预测概率
test_pred_class = lr_meta.predict(meta_test)  # 预测类别（默认阈值0.5）

# 计算评估指标
def evaluate_model(y_true, y_pred, y_prob):
    """计算分类模型的各项评估指标"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print("测试集堆叠模型评估结果：")
    print(f"混淆矩阵:\n{cm}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"Kappa值: {kappa:.4f}")
    return cm, accuracy, precision, recall, f1, kappa

# 评估堆叠模型在测试集上的表现
stacking_metrics = evaluate_model(TestY, test_pred_class, test_pred_prob)
print(stacking_metrics)


# %% [markdown]
# # 预测

# %%
new_data = pd.read_excel("C:\\Users\\31597\\Desktop\\Traditional Chinese Medicine\\Wang TCM\\2300预测数据集.xlsx",sheet_name=2) 
new_data = new_data[X.columns]
print(new_data)

# %%
rf_pred = best_rf.predict(new_data)
xgb_pred = best_xgb.predict(new_data)
gbm_pred = best_gbm.predict(new_data)
ada_pred = best_ada.predict(new_data)
ann_pred = best_ann.predict(new_data)

# 将预测结果添加到原始数据中
new_data['RandomForest_Prediction'] = rf_pred
new_data['XGBoost_Prediction'] = xgb_pred
new_data['HistGradientBoosting_Prediction'] = gbm_pred
new_data['AdaBoost_Prediction'] = ada_pred
new_data['ANN_Prediction'] = ann_pred

# 将结果保存到新的 Excel 文件中
new_data.to_excel("prediction_results.xlsx", index=False)


