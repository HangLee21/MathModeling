# import warnings
#
# import numpy as np
#
# warnings.filterwarnings("ignore")
#
# import pandas as pd
# from lightgbm import LGBMClassifier
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
#
# # 读取 Excel 文件
# df = pd.read_excel('../data/standard_data.xlsx')
# columns = df.columns[:-1]
#
#
# def model_functoion(model):
#     auc = round(cross_val_score(model, df[columns].values, df['label'].values, cv=5, scoring='roc_auc').mean(), 2)
#     acc = round(cross_val_score(model, df[columns].values, df['label'].values, cv=5, scoring='accuracy').mean(), 2)
#     recall = round(cross_val_score(model, df[columns].values, df['label'].values, cv=5, scoring='recall').mean(), 2)
#     precision = round(cross_val_score(model, df[columns].values, df['label'].values, cv=5, scoring='precision').mean(),
#                       2)
#     f1 = round(cross_val_score(model, df[columns].values, df['label'].values, cv=5, scoring='f1').mean(), 2)
#     return auc, acc, recall, precision, f1
#
#
# model = LGBMClassifier(random_state=30)
# print("LGBM:", model_functoion(model))
# model = XGBClassifier()
# print("XGB:", model_functoion(model))
# model = SVC(random_state=50)
# print("SVC:", model_functoion(model))
# model = MLPClassifier(random_state=50)
# print("MLP:", model_functoion(model))
# model = LogisticRegression(random_state=50)
# print("Logist:", model_functoion(model))
#
# def draw_predict(model_ls, name_ls, types = 'train'):
#     plt.figure(figsize=(8, 7), dpi=80, facecolor='w')
#     plt.xlim((-0.01, 1.02))
#     plt.ylim((-0.01, 1.02))
#     plt.xticks(np.arange(0, 1.1, 0.1))
#     plt.yticks(np.arange(0, 1.1, 0.1))
#
#
#     if types == 'test':
#         for model, name in zip(model_ls, name_ls):
#             ytest = model.predict_proba()
#     else:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# 读取 Excel 文件
dataset = pd.read_excel('../data/standard_data.xlsx')
columns = dataset.columns[:-1]

# 假设你的特征数据保存在 X 中，标签数据保存在 y 中
X_train, X_test, y_train, y_test = train_test_split(dataset[columns].values, dataset['label'].values, test_size=0.2,
                                                    random_state=42)

# 创建并训练模型
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

svc_model = SVC(probability=True, max_iter=2000)
svc_model.fit(X_train, y_train)

mlp_model = MLPClassifier(max_iter=2000)
mlp_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)

# 在测试集上进行预测
lgb_y_pred_prob = lgb_model.predict_proba(X_test)[:, 1]
xgb_y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
svc_y_pred_prob = svc_model.predict_proba(X_test)[:, 1]
mlp_y_pred_prob = mlp_model.predict_proba(X_test)[:, 1]
lr_y_pred_prob = lr_model.predict_proba(X_test)[:, 1]

# 计算指标
lgb_fpr, lgb_tpr, lgb_thresholds = roc_curve(y_test, lgb_y_pred_prob)
lgb_roc_auc = auc(lgb_fpr, lgb_tpr)
lgb_accuracy = accuracy_score(y_test, lgb_model.predict(X_test))
lgb_recall = recall_score(y_test, lgb_model.predict(X_test))
lgb_precision = precision_score(y_test, lgb_model.predict(X_test))
lgb_f1 = f1_score(y_test, lgb_model.predict(X_test))

xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_y_pred_prob)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
xgb_accuracy = accuracy_score(y_test, xgb_model.predict(X_test))
xgb_recall = recall_score(y_test, xgb_model.predict(X_test))
xgb_precision = precision_score(y_test, xgb_model.predict(X_test))
xgb_f1 = f1_score(y_test, xgb_model.predict(X_test))

svc_fpr, svc_tpr, svc_thresholds = roc_curve(y_test, svc_y_pred_prob)
svc_roc_auc = auc(svc_fpr, svc_tpr)
svc_accuracy = accuracy_score(y_test, svc_model.predict(X_test))
svc_recall = recall_score(y_test, svc_model.predict(X_test))
svc_precision = precision_score(y_test, svc_model.predict(X_test))
svc_f1 = f1_score(y_test, svc_model.predict(X_test))

mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_y_pred_prob)
mlp_roc_auc = auc(mlp_fpr, mlp_tpr)
mlp_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))
mlp_recall = recall_score(y_test, mlp_model.predict(X_test))
mlp_precision = precision_score(y_test, mlp_model.predict(X_test))
mlp_f1 = f1_score(y_test, mlp_model.predict(X_test))

lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_y_pred_prob)
lr_roc_auc = auc(lr_fpr, lr_tpr)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
lr_recall = recall_score(y_test, lr_model.predict(X_test))
lr_precision = precision_score(y_test, lr_model.predict(X_test))
lr_f1 = f1_score(y_test, lr_model.predict(X_test))

# 绘制 ROC 曲线
plt.figure()
plt.plot(lgb_fpr, lgb_tpr, label='LGBM (AUC = %0.2f)' % lgb_roc_auc)
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost (AUC = %0.2f)' % xgb_roc_auc)
plt.plot(svc_fpr, svc_tpr, label='SVC (AUC =%0.2f)' % svc_roc_auc)
plt.plot(mlp_fpr, mlp_tpr, label='MLP (AUC = %0.2f)' % mlp_roc_auc)
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression (AUC = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 打印指标结果
print("LGBM:")
print("AUC:", lgb_roc_auc)
print("Accuracy:", lgb_accuracy)
print("Recall:", lgb_recall)
print("Precision:", lgb_precision)
print("F1 Score:", lgb_f1)
print()

print("XGBoost:")
print("AUC:", xgb_roc_auc)
print("Accuracy:", xgb_accuracy)
print("Recall:", xgb_recall)
print("Precision:", xgb_precision)
print("F1 Score:", xgb_f1)
print()

print("SVC:")
print("AUC:", svc_roc_auc)
print("Accuracy:", svc_accuracy)
print("Recall:", svc_recall)
print("Precision:", svc_precision)
print("F1 Score:", svc_f1)
print()

print("MLP:")
print("AUC:", mlp_roc_auc)
print("Accuracy:", mlp_accuracy)
print("Recall:", mlp_recall)
print("Precision:", mlp_precision)
print("F1 Score:", mlp_f1)
print()

print("Logistic Regression:")
print("AUC:", lr_roc_auc)
print("Accuracy:", lr_accuracy)
print("Recall:", lr_recall)
print("Precision:", lr_precision)
print("F1 Score:", lr_f1)

df = pd.read_csv(r"../data/Wimbledon_featured_matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)
index = df[df.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
test = dataset.iloc[index]
train = dataset.drop(index, axis=0)
wins = test['label'].tolist()
lgb_model.fit(train[columns].values, train['label'].values)
pred = lgb_model.predict_proba(test[columns].values)

pred = pd.DataFrame({
    "score:": pred[:, 0],
    "win": wins,
})

match = [pred.iloc[:45], pred.iloc[45:126], pred.iloc[126:195], pred.iloc[195:259], pred.iloc[259:]]

plt.figure(figsize=(48, 6), dpi=80, facecolor='w')
plt.xlabel('Points')
plt.ylabel('Momentum')
for i in range(len(match)):
    plt.plot(match[i].index, match[i].values)
plt.show()

f = pd.DataFrame({
    'col': columns.tolist(),
    'score': lgb_model.feature_importances_
}).sort_values(by='score')
print(f)

y_pred = lgb_model.predict(test[columns].values)
# 将预测结果转换为类别标签（如果是分类问题）
y_pred_class = np.round(y_pred)

# 绘制预测结果
y_test = test['label']
plt.scatter(range(len(y_test)), y_test, label='Actual')
plt.scatter(range(len(y_test)), y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Target')
plt.title('Prediction Results')
plt.legend()
plt.show()

# 计算准确率
accuracy = accuracy_score(y_test, y_pred_class)
print('Accuracy:', accuracy)

