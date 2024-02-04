import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 假设已经加载了比赛数据到DataFrame `df`
df = pd.read_csv(r"../data/Wimbledon_featured_matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)
df = df[df.match_id == '2023-wimbledon-1701']
# 示例：为简化，这里假设数据已经过滤到特定比赛和发球权
# 初始化势头分数
df['momentum_score_1'] = 0
df['momentum_score_2'] = 0
df['momentum_score_diff'] = 0
momentum_score = 0
consecutive_wins = 0
for index, row in df.iterrows():
    P_t = 1 if row['point_victor'] == 1 else 0
    S_t = 1.2 if row['server'] == 1 else 1.0 # 假设player_name是发球方
    if P_t == 1:
        consecutive_wins += 1
    else:
        consecutive_wins = 0
    C_t = 1 + consecutive_wins * 0.2
    momentum_score += (P_t * S_t * C_t)
    df.at[index, 'momentum_score_1'] = momentum_score
momentum_score = 0
for index, row in df.iterrows():
    P_t = 1 if row['point_victor'] == 2 else 0
    S_t = 1.2 if row['server'] == 2 else 1.0 # 假设player_name是发球方
    if P_t == 1:
        consecutive_wins += 1
    else:
        consecutive_wins = 0
    C_t = 1 + consecutive_wins * 0.2
    momentum_score += (P_t * S_t * C_t)
    df.at[index, 'momentum_score_2'] = momentum_score
print(df['momentum_score_1'])
print(df['momentum_score_2'])
# 可视化势头分数变化
for index, row in df.iterrows():
    df.at[index, 'momentum_score_diff'] = df.at[index, 'momentum_score_1'] - df.at[index, 'momentum_score_2']
print(df['momentum_score_diff'])
df['gradient'] = (df['momentum_score_diff'] > df['momentum_score_diff'].shift(1)).astype(int)
df['gradient'].fillna(0, inplace=True)
row_data = df["momentum_score_diff"]
df['point_victor'] = -df['point_victor'] + 2
scaler = MinMaxScaler()
normalized_row_data = scaler.fit_transform(row_data.values.reshape(-1, 1))
plt.figure(figsize=(48, 6))
plt.plot(df['point_no'], df['gradient'], label='Momentum Score')
plt.plot(df['point_no'], df['point_victor'], label='Really Score')
plt.xlabel('Point Number')
plt.ylabel('Momentum Score')
plt.title('Match Momentum Flow')
plt.legend()
plt.show()