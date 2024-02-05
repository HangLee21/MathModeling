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

# 创建负载图
plt.figure(figsize=(18, 6))
plt.imshow(factor_loadings.T, cmap='coolwarm', aspect='auto')
plt.colorbar()

# 设置坐标轴标签和标题
plt.xlabel('Features')
plt.ylabel('Principal Components')
plt.title('Loading Plot')
n_features = 12
features = [
    'set_win',
    'ace',
    'server',
    'speed',
    'run',
    'rally_count',
    'AD_score',
    'continue_wins',
    'unforced_error',
    'break_rate',
    'net_rate',
    'serve_rate'
]
# 设置x轴刻度标签
plt.xticks(range(n_features), features)
pcs = []
for i in range(num_components):
    pcs.append(f'PC {i}')
# 设置y轴刻度标签
plt.yticks(range(num_components), pcs)

# 显示负载图
plt.show()

# 计算方差解释率
variance_explained = fa.get_factor_variance()

# 打印方差解释率
print("\n方差解释率：")
print(variance_explained)

# 计算判断矩阵
factor_loadings = factor_loadings.T

# 创建4x4的初始判断矩阵
judgment_matrix = np.array(
                    [[1, 1/3, 1/5, 1/7],
                   [3, 1, 1/3, 1/5],
                   [5, 3, 1, 1/3],
                   [7, 5, 3, 1]])

# 计算特征向量和特征值
eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)

# 获取最大特征值
max_eigenvalue = max(eigenvalues)

# 计算一致性指标CI
n = judgment_matrix.shape[0]
ci = (max_eigenvalue - n) / (n - 1)

# 预定义的随机一致性指标RI表
ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

# 查找随机一致性指标RI
ri = ri_table[n]

# 计算一致性比率CR
cr = ci / ri

# 输出CI、RI和CR值
print("CI值：", ci)
print("RI值：", ri)
print("CR值：", cr)

# 计算特征向量
eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

# 归一化处理
weight_vector = principal_eigenvector / np.sum(principal_eigenvector)

# 输出结果
print("判断矩阵的权重向量:")
print(weight_vector.real)
# 计算判断矩阵的特征向量和权重

weights = weight_vector
# 构建决策矩阵
decision_matrix = np.array([[0.8, 0.6, 0.7, 0.9],
                            [0.7, 0.9, 0.6, 0.8],
                            [0.9, 0.7, 0.8, 0.6]])

# 标准化决策矩阵
normalized_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))

# 确定正理想解和负理想解
positive_ideal_solution = np.max(normalized_matrix, axis=0)
negative_ideal_solution = np.min(normalized_matrix, axis=0)

# 计算距离
positive_distances = np.sqrt(np.sum((normalized_matrix - positive_ideal_solution)**2, axis=1))
negative_distances = np.sqrt(np.sum((normalized_matrix - negative_ideal_solution)**2, axis=1))

# 计算接近程度
closeness = negative_distances / (positive_distances + negative_distances)

# 输出结果
print("权重：", weights)
print("一致性比率：", cr)
print("接近程度：", closeness)


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

bg_color = 'lightblue'  # 背景颜色
ranges = [[0, 46], [46, 127], [127, 196], [196, 260], [259, 334]]

plt.figure(figsize=(24, 6), dpi=80, facecolor='w')
plt.plot(df['point_no'], p_ls, label='P1 Performance Score')
plt.plot(df['point_no'], p_ls2, label='P2 Performance Score')
# plt.plot(df['point_no'], df['point_victor'], label='Really Score')
# plt.plot(df['point_no'], df['server'], label='Server')
for i in range(5):
    if i == 1 or i == 2 or i == 4:
        # 绘制特定范围的背景颜色
        plt.axvspan(ranges[i][0], ranges[i][1], facecolor=bg_color, alpha=0.3)
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.show()

# p1 win game 2 3 5
x_labels =[df['point_no'].iloc[:45], df['point_no'].iloc[45:126], df['point_no'].iloc[126:195], df['point_no'].iloc[195:259], df['point_no'].iloc[259:]]
match_1 = [p_ls[:45], p_ls[45:126], p_ls[126:195], p_ls[195:259], p_ls[259:]]
plt.figure(figsize=(24, 6), dpi=80)
plt.xlabel('Points')
plt.ylabel('Player1 Momentum')
for i in range(len(match_1)):
    plt.plot(x_labels[i], match_1[i])
    if i == 1 or i == 2 or i == 4:
        # 绘制特定范围的背景颜色
        plt.axvspan(ranges[i][0], ranges[i][1], facecolor=bg_color, alpha=0.3)
plt.show()

match_2 = [p_ls2[:45], p_ls2[45:126], p_ls2[126:195], p_ls2[195:259], p_ls2[259:]]
plt.figure(figsize=(48, 6), dpi=80)
plt.xlabel('Points')
plt.ylabel('Player2 Momentum')
for i in range(len(match_1)):
    plt.plot(x_labels[i], match_2[i])
    if i == 0 or i == 3:
        # 绘制特定范围的背景颜色
        plt.axvspan(ranges[i][0], ranges[i][1], facecolor=bg_color, alpha=0.3)
plt.show()


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
df['momentum_diff'] = df['Momentum'] - df2['Momentum']
df2['momentum_diff'] = df2['Momentum'] - df['Momentum']
df['sign_reversal'] = 0
# 迭代每个元素
for i in range(len(df) - 5):
    current_momentum = df.loc[i, 'momentum_diff']
    next_momentum = df.loc[i + 1:i + 5, 'momentum_diff']

    # 检查正负号变化
    if (next_momentum > 0).all() or (next_momentum < 0).all():
        df.loc[i, 'sign_reversal'] = 0
    else:
        df.loc[i, 'sign_reversal'] = 1

df2['sign_reversal'] = df['sign_reversal']

df.to_excel('../data/momentum_1.xlsx', index=False)
df2.to_excel('../data/momentum_2.xlsx', index=False)

Woman = True
if Woman:
    ##### 存储女子势头分数
    # 读取 Excel 文件
    df = pd.read_excel('../data/woman_pca_standard_data_1.xlsx')
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

    df2 = pd.read_excel('../data/woman_pca_standard_data_2.xlsx')
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
    df['momentum_diff'] = df['Momentum'] - df2['Momentum']
    df2['momentum_diff'] = df2['Momentum'] - df['Momentum']
    df['sign_reversal'] = 0
    # 迭代每个元素
    for i in range(len(df) - 5):
        current_momentum = df.loc[i, 'momentum_diff']
        next_momentum = df.loc[i + 1:i + 5, 'momentum_diff']

        # 检查正负号变化
        if (next_momentum > 0).all() or (next_momentum < 0).all():
            df.loc[i, 'sign_reversal'] = 0
        else:
            df.loc[i, 'sign_reversal'] = 1

    df2['sign_reversal'] = df['sign_reversal']

    df.to_excel('../data/woman_momentum_1.xlsx', index=False)
    df2.to_excel('../data/woman_momentum_2.xlsx', index=False)

    index = df[df.match_id == '2023-wimbledon-2101'].reset_index(drop=True).index
    df = df.iloc[index]

    index = df2[df2.match_id == '2023-wimbledon-2101'].reset_index(drop=True).index
    df2 = df2.iloc[index]

    df3= pd.read_csv(r"../data/2023-wimbledon-female-matches.csv")
    df3.loc[(df3.p1_score == 'AD', 'p1_score')] = 50
    df3.loc[(df3.p2_score == 'AD', 'p2_score')] = 50
    df3['p1_score'] = df3['p1_score'].astype(int)
    df3['p2_score'] = df3['p2_score'].astype(int)
    # 删除缺省值
    df3.dropna(subset=['speed_mph'], inplace=True)
    df3.drop(df3[(df3['point_no'] == '0X') | (df3['point_no'] == '0Y')].index, inplace=True)
    df3['point_no'] = df3['point_no'].astype(int)
    df3 = df3[df3.match_id == '2023-wimbledon-2101']

    plt.figure(figsize=(24, 6), dpi=80, facecolor='w')
    plt.plot(df3['point_no'], df['Momentum'], label='P1 Performance Score')
    plt.plot(df3['point_no'], df2['Momentum'], label='P2 Performance Score')
    plt.xlabel('Point Number')
    plt.ylabel('Momentum Score')
    plt.title('Match Momentum Flow')
    plt.legend()
    plt.show()

