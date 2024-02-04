import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from scipy.stats import kruskal, pearsonr
# 读取 Excel 文件
df = pd.read_excel('../data/pca_standard_data_1.xlsx')
columns = df.columns[:-2]
data = pd.DataFrame(df[columns].values)

# 检查DataFrame中是否存在NaN
has_nan = data.isna().any().any()

# 检查DataFrame中是否存在Inf
has_inf = data.isin([np.inf, -np.inf]).any().any()

# 判断是否存在无效值
has_invalid_values = has_nan or has_inf

if has_invalid_values:
    print("数据中存在无效值")
else:
    print("数据中没有无效值")
kmo = calculate_kmo(data)
bartlett = calculate_bartlett_sphericity(data)
print(f'KMO:{kmo[1]}')
print(f'Bartlett:{bartlett[1]}')
# PCA
pca = PCA()
principal_components = pca.fit_transform(data)

# 计算方差解释比例
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 绘制 scree plot
num_components = len(explained_variance_ratio)
plt.plot(range(1, num_components + 1), explained_variance_ratio, marker='o')
plt.xlabel('Component')
plt.ylabel('Variance Explained (%)')
plt.title('Scree Plot')
plt.xticks(range(1, num_components + 1))
plt.show()

# 计算累计方差解释率
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# 确定需要多少个主成分
num_components = np.argmax(explained_variance_ratio >= 0.8) + 1

print("需要的主成分数量:", num_components)

# 创建因子分析对象
fa = FactorAnalyzer(n_factors=num_components, rotation='varimax')

# 执行因子分析
fa.fit(data)

# 计算因子载荷
factor_loadings = fa.loadings_

# 打印因子载荷
print("因子载荷矩阵：")
print(factor_loadings)

# 计算方差解释率
variance_explained = fa.get_factor_variance()

# 打印方差解释率
print("\n方差解释率：")
print(variance_explained)

# 计算判断矩阵
factor_loadings = factor_loadings.T
num_factors = len(factor_loadings)
judgment_matrix = np.zeros((num_factors, num_factors))

for i in range(num_factors):
    for j in range(num_factors):
        if i == j:
            judgment_matrix[i, j] = 1
        elif i < j:
            judgment_matrix[i, j] = factor_loadings[i, :].sum() / factor_loadings[j, :].sum()
        else:
            judgment_matrix[i, j] = factor_loadings[i, :].sum() / factor_loadings[j, :].sum()

print("判断矩阵:")
print(judgment_matrix)

# 计算特征向量
column_sum = np.sum(judgment_matrix, axis=0)
normalized_matrix = judgment_matrix / column_sum
feature_vector = np.mean(normalized_matrix, axis=1)
# 计算一致性指标 CI
n = len(judgment_matrix)
lambda_max = np.sum(column_sum * feature_vector)
CI = (lambda_max - n) / (n - 1)

# 计算一致性比率 RI
RI_table = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
RI = RI_table[n - 1]

# 计算一致性比率 CR
CR = CI / RI

print("CI:", CI)
print("RI:", RI)
print("CR:", CR)

# 计算特征向量
eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

# 归一化处理
weight_vector = principal_eigenvector / np.sum(principal_eigenvector)

# 输出结果
print("判断矩阵的权重向量:")
print(weight_vector.real)
df = df[df.match_id == '2023-wimbledon-1701']
data = df[columns]
p_ls = []
for index, row in data.iterrows():
    p = 0
    for i in range(len(factor_loadings)):
        F = factor_loadings[i]
        w = 0
        for j in range(len(F)):
            w += F[j]*row.values[j]
        p += w*weight_vector.real.tolist()[i]
    p_ls.append(p)

df2 = pd.read_excel('../data/pca_standard_data_2.xlsx')
columns2 = df2.columns[:-2]

df2 = df2[df2.match_id == '2023-wimbledon-1701']
data2 = df2[columns]
p_ls2 = []
for index, row in data2.iterrows():
    p = 0
    for i in range(len(factor_loadings)):
        F = factor_loadings[i]
        w = 0
        for j in range(len(F)):
            w += F[j]*row.values[j]
        p += w*weight_vector.real.tolist()[i]
    p_ls2.append(p)

df = pd.read_csv(r"../data/Wimbledon_featured_matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)
df = df[df.match_id == '2023-wimbledon-1701']
df['point_victor'] = -df['point_victor'] + 2

plt.figure(figsize=(48, 6))
plt.plot(df['point_no'], p_ls, label='P1 Performance Score')
plt.plot(df['point_no'], p_ls2, label='P2 Performance Score')
# plt.plot(df['point_no'], df['point_victor'], label='Really Score')
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.show()
print(p_ls2)
accuracy = 0
for i in range(len(p_ls)):
    if p_ls[i] > p_ls2[i] and df['point_victor'].values[i] == 1:
        accuracy = accuracy + 1
    elif p_ls[i] < p_ls2[i] and df['point_victor'].values[i] == 0:
        accuracy = accuracy + 1

accuracy = accuracy / len(p_ls)
print('Accuracy:', accuracy)

p_ls_diff = [a - b for a, b in zip(p_ls, p_ls2)]

# 执行 Kruskal-Wallis H 检验
statistic, p_value = kruskal(p_ls_diff, df['point_victor'].values)

# 输出结果
print("Kruskal-Wallis H 检验结果:")
print("统计量 (H):", statistic)
print("p 值:", p_value)

# 计算皮尔逊相关系数和p值
correlation, p_value = pearsonr(p_ls_diff, df['point_victor'].values)
# 输出结果
print("皮尔逊相关系数:", correlation)
print("p 值:", p_value)

##### 存储势头分数
# 读取 Excel 文件
df = pd.read_excel('../data/pca_standard_data_1.xlsx')
columns = df.columns[:-2]
data = df[columns]
for index, row in data.iterrows():
    p = 0
    for i in range(len(factor_loadings)):
        F = factor_loadings[i]
        w = 0
        for j in range(len(F)):
            w += F[j]*row.values[j]
        p += w*weight_vector.real.tolist()[i]
    df.at[index, 'Momentum'] = p
df.to_excel('../data/momentum_1.xlsx', index=False)

df2 = pd.read_excel('../data/pca_standard_data_2.xlsx')
columns2 = df2.columns[:-2]
data2 = df2[columns2]
for index, row in data2.iterrows():
    p = 0
    for i in range(len(factor_loadings)):
        F = factor_loadings[i]
        w = 0
        for j in range(len(F)):
            w += F[j]*row.values[j]
        p += w*weight_vector.real.tolist()[i]
    df2.at[index, 'Momentum'] = p
df2.to_excel('../data/momentum_2.xlsx', index=False)

