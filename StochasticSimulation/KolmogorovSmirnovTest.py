import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, binom_test

# 假设df是包含比赛数据的DataFrame
# 假设已经计算了实际比赛的势头得分并存储在'momentum_score'列
# 随机比赛模拟函数

df = pd.read_excel('../data/momentum_1.xlsx')
index = df[df.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
df = df.iloc[index]

df2 = pd.read_excel('../data/momentum_2.xlsx')
index = df2[df2.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
df2 = df2.iloc[index]

dataset = pd.read_csv(r"../data/Wimbledon_featured_matches.csv")
dataset.loc[(dataset.p1_score == 'AD', 'p1_score')] = 50
dataset.loc[(dataset.p2_score == 'AD', 'p2_score')] = 50
dataset['p1_score'] = dataset['p1_score'].astype(int)
dataset['p2_score'] = dataset['p2_score'].astype(int)
# 删除缺省值
dataset.dropna(subset=['speed_mph'], inplace=True)
index = dataset[dataset.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
dataset = dataset.iloc[index]

def simulate_random_match(len):
    # for match_id, set_no, game_no, point_no in zip(dataset.match_id, dataset.set_no, dataset.game_no, dataset.point_no):
    #     now_match = dataset[dataset.match_id == match_id]
    #     now_set = now_match[now_match.set_no == set_no]
    #     now_game = now_set[now_set.game_no == game_no]
    #     now_point = now_game[now_game.point_no == point_no]
    #     if now_point['server'].values[0] == 1:
    #         serve_win_prob = 0.5
    #     else:
    #         serve_win_prob = 0.5
    #     # 根据概率生成每个比赛点的胜负情况
    #     win = 1 if np.random.rand(1) < serve_win_prob else 0
    serve_win_prob = 0.5
    win = np.random.rand(len) < serve_win_prob
    win = np.array(win).astype(int)
    return win


# 实际比赛的势头得分
actual_momentum_scores_diff = df['Momentum'] - df2['Momentum']

# 模拟N次随机比赛
N = 1000
simulated_scores = [np.cumsum(simulate_random_match(len(actual_momentum_scores_diff))) for _ in range(N)]

actual_momentum_scores_diff = np.where(actual_momentum_scores_diff > 0, 1, 0)

# ks_stat, p_value = stats.ks_2samp(np.cumsum(actual_momentum_scores_diff), np.concatenate(simulated_scores))
# print(f"KS statistic: {ks_stat}, P-value: {p_value}")
# # 可视化实际与模拟的势头得分分布
# plt.hist(actual_momentum_scores_diff, bins=30, alpha=0.5, label='Actual')
# plt.hist(simulated_scores, bins=30, alpha=0.5, label='Simulated')
# # 设置横坐标刻度和标签
# plt.xlabel('Momentum Score')
# plt.ylabel('Frequency')
# plt.legend()
# plt.title('Distribution of Momentum Scores: Actual vs. Simulated')
# plt.show()

p_values = []
correlations = []
for i in range(1000):
    # 计算皮尔逊相关系数和p值
    s_test = simulate_random_match(len(actual_momentum_scores_diff))
    correlation, p_value = pearsonr(actual_momentum_scores_diff, s_test)
    # 输出结果
    print(correlation)
    correlations.append(correlation)
    p_values.append(p_value)

print('相关系数: ', np.mean(correlations))
print("p 值:", np.mean(p_values))

