import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"../data/2023-wimbledon-female-matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)
df.drop(df[(df['point_no'] == '0X') | (df['point_no'] == '0Y')].index, inplace=True)
df['point_no'] = df['point_no'].astype(int)
df.to_excel('../data/woman_deleted_data.xlsx', index=False)
x_ls = []
labels = []
match_ids = []
x_length = 13
for i in range(x_length):
    x_ls.append([])

for match_id, set_no, game_no, point_no in zip(df.match_id, df.set_no, df.game_no, df.point_no):
    now_match = df[df.match_id == match_id]
    now_set = now_match[now_match.set_no == set_no]
    now_game = now_set[now_set.game_no == game_no]
    now_point = now_game[now_game.point_no == point_no]
    # x0 当前set领先局
    # x1 ace球
    # x2 是否发球
    # x3 球速
    # x4 跑动距离
    # x5 回击次数
    # x6 领先分数
    # x7 本game连续得分或失分局数
    # x8 unforced error
    # x9 上一个set的破局率
    # x10 当前领先set
    # x11 上一个game 上网次数与上网得分比例
    # x12 单发失误率
    now_x_ls = []
    # x0
    now_x_ls.append(now_point['p1_games'].values[0] - now_point['p2_games'].values[0])
    # x1
    now_x_ls.append(1 if now_point['p1_ace'].values[0] == 1 else 0)
    # x2
    now_x_ls.append(1 if now_point['server'].values[0] == 1 else 0)
    # x3
    now_x_ls.append(now_point['speed_mph'].values[0] if now_point['server'].values[0] == 1 else 0)
    # x4
    now_x_ls.append(now_point['p1_distance_run'].values[0])
    # x5
    now_x_ls.append(now_point['rally_count'].values[0])
    # x6
    now_x_ls.append(now_point['p1_score'].values[0] - now_point['p2_score'].values[0])
    # x7
    wins = 0
    point_prev_no = now_point.point_no.values[0] - 1
    prev_point = now_game[now_game.point_no == point_prev_no]
    while len(prev_point) != 0:
        # 开始
        if wins == 0:
            if prev_point['point_victor'].values[0] == 1:
                wins += 1
            else:
                wins -= 1
        # 丢分
        elif wins < 0:
            if prev_point['point_victor'].values[0] != 1:
                wins -= 1
            else:
                break
        # 得分
        elif wins > 0:
            if prev_point['point_victor'].values[0] == 1:
                wins += 1
            else:
                break
        point_prev_no = prev_point.point_no.values[0] - 1
        prev_point = now_game[now_game.point_no == point_prev_no]
    now_x_ls.append(wins)
    # x8
    now_x_ls.append(1 if now_point['p1_unf_err'].values[0] == 1 else 0)
    # x9
    now_x_ls.append(
        now_set['p1_break_pt_won'].sum() / now_set['p1_break_pt'].sum() if now_set['p1_break_pt'].sum() != 0 else 0)
    # break_rate = 0
    # set_prev_no = now_set.set_no.values[0] - 1
    # prev_set = now_match[now_match.set_no == set_prev_no]
    # if len(prev_set) == 0:
    #     break_rate = 0
    # else:
    #     break_rate = prev_set['p1_break_pt_won'].sum() / prev_set['p1_break_pt'].sum() if prev_set['p1_break_pt'].sum() != 0 else 0
    # now_x_ls.append(break_rate)
    # x10
    now_x_ls.append(0)
    # x11
    now_x_ls.append(
        now_game['p1_net_pt_won'].sum() / now_game['p1_net_pt'].sum() if now_game['p1_net_pt'].sum() != 0 else 0)
    # net_rate = 0
    # game_prev_no = now_game.game_no.values[0] - 1
    # prev_game = now_set[now_set.game_no == game_prev_no]
    # if len(prev_game) == 0:
    #     net_rate = 0
    # else:
    #     net_rate = prev_game['p1_net_pt_won'].sum()/ prev_game['p1_net_pt'].sum() if prev_game['p1_net_pt'].sum() != 0 else 0
    # now_x_ls.append(net_rate)
    # x12
    count_ones = (now_game['serve_no'] == 1).sum()
    total_count = len(now_game)
    proportion = count_ones / total_count
    now_x_ls.append(proportion if now_point['server'].values[0] == 1 else 0 )
    # label
    labels.append(1 if now_point['point_victor'].values[0] == 1 else 0)
    # id
    match_ids.append(now_point['match_id'].values[0])
    for i in range(len(now_x_ls)):
        x_ls[i].append(now_x_ls[i])

dataset = pd.DataFrame({
    'set_win': x_ls[0],
    'ace': x_ls[1],
    'server': x_ls[2],
    'speed': x_ls[3],
    'run': x_ls[4],
    'rally_count': x_ls[5],
    'AD_score': x_ls[6],
    'continue_wins': x_ls[7],
    'unforced_error': x_ls[8],
    'break_rate': x_ls[9],
    'net_rate': x_ls[11],
    'serve_rate': x_ls[12],
    'label': labels,
    'match_id': match_ids,
})
dataset.to_excel('../data/woman_pca_origin_data_1.xlsx', index=False)
scaler = MinMaxScaler()
columns = dataset.columns[:-2]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_excel('../data/woman_pca_standard_data_1.xlsx', index=False)


df = pd.read_csv(r"../data/2023-wimbledon-female-matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)
df.drop(df[(df['point_no'] == '0X') | (df['point_no'] == '0Y')].index, inplace=True)
df['point_no'] = df['point_no'].astype(int)
df.to_excel('../data/woman_deleted_data.xlsx', index=False)

x_ls = []
labels = []
match_ids = []
for i in range(x_length):
    x_ls.append([])

for match_id, set_no, game_no, point_no in zip(df.match_id, df.set_no, df.game_no, df.point_no):
    now_match = df[df.match_id == match_id]
    now_set = now_match[now_match.set_no == set_no]
    now_game = now_set[now_set.game_no == game_no]
    now_point = now_game[now_game.point_no == point_no]
    last_point = now_game[now_game.point_no == point_no - 1]
    # x0 当前set领先game
    # x1 ace球
    # x2 是否发球
    # x3 球速
    # x4 跑动距离
    # x5 回击次数
    # x6 领先分数
    # x7 本game连续得分或失分局数
    # x8 是否存在unforced error
    now_x_ls = []
    # x0
    now_x_ls.append(now_point['p2_games'].values[0] - now_point['p1_games'].values[0])
    # x1
    now_x_ls.append(1 if now_point['p2_ace'].values[0] == 1 else 0)
    # x2
    now_x_ls.append(1 if now_point['server'].values[0] == 2 else 0)
    # x3
    now_x_ls.append(now_point['speed_mph'].values[0] if now_point['server'].values[0] == 2 else 0)
    # x4
    now_x_ls.append(now_point['p2_distance_run'].values[0])
    # x5
    now_x_ls.append(now_point['rally_count'].values[0])
    # x6
    now_x_ls.append(now_point['p2_score'].values[0] - now_point['p1_score'].values[0])
    # x7
    wins = 0
    point_prev_no = now_point.point_no.values[0] - 1

    prev_point = now_game[now_game.point_no == point_prev_no]
    while len(prev_point) != 0:
        # 开始
        if wins == 0:
            if prev_point['point_victor'].values[0] == 2:
                wins += 1
            else:
                wins -= 1
        # 丢分
        elif wins < 0:
            if prev_point['point_victor'].values[0] != 2:
                wins -= 1
            else:
                break
        # 得分
        elif wins > 0:
            if prev_point['point_victor'].values[0] == 2:
                wins += 1
            else:
                break
        point_prev_no = prev_point.point_no.values[0] - 1
        prev_point = now_game[now_game.point_no == point_prev_no]
    now_x_ls.append(wins)
    # x8
    now_x_ls.append(1 if now_point['p2_unf_err'].values[0] == 1 else 0)
    # x9
    now_x_ls.append(
        now_set['p2_break_pt_won'].sum() / now_set['p2_break_pt'].sum() if now_set['p2_break_pt'].sum() != 0 else 0)
    # break_rate = 0
    # set_prev_no = now_set.set_no.values[0] - 1
    # prev_set = now_match[now_match.set_no == set_prev_no]
    # if len(prev_set) == 0:
    #     break_rate = 0
    # else:
    #     break_rate = prev_set['p1_break_pt_won'].sum() / prev_set['p1_break_pt'].sum() if prev_set['p1_break_pt'].sum() != 0 else 0
    # now_x_ls.append(break_rate)
    # x10
    now_x_ls.append(0)
    # x11
    now_x_ls.append(
        now_game['p2_net_pt_won'].sum() / now_game['p2_net_pt'].sum() if now_game['p2_net_pt'].sum() != 0 else 0)
    # net_rate = 0
    # game_prev_no = now_game.game_no.values[0] - 1
    # prev_game = now_set[now_set.game_no == game_prev_no]
    # if len(prev_game) == 0:
    #     net_rate = 0
    # else:
    #     net_rate = prev_game['p1_net_pt_won'].sum()/ prev_game['p1_net_pt'].sum() if prev_game['p1_net_pt'].sum() != 0 else 0
    # now_x_ls.append(net_rate)
    # x12
    count_ones = (now_game['serve_no'] == 1).sum()
    total_count = len(now_game)
    proportion = count_ones / total_count
    now_x_ls.append(proportion if now_point['server'].values[0] == 2 else 0)
    # # x9
    # break_rate = 0
    # set_prev_no = now_set.set_no.values[0] - 1
    # prev_set = now_match[now_match.set_no == set_prev_no]
    # if len(prev_set) == 0:
    #     break_rate = 0
    # else:
    #     break_rate = prev_set['p2_break_pt_won'].sum() / prev_set['p2_break_pt'].sum() if prev_set['p2_break_pt'].sum() != 0 else 0
    # now_x_ls.append(break_rate)
    # # x10
    # now_x_ls.append(now_point['p2_sets'].values[0] - now_point['p1_sets'].values[0])
    # # x11
    # net_rate = 0
    # game_prev_no = now_game.game_no.values[0] - 1
    # prev_game = now_set[now_set.game_no == game_prev_no]
    # if len(prev_game) == 0:
    #     net_rate = 0
    # else:
    #     net_rate = prev_game['p2_net_pt_won'].sum() / prev_game['p2_net_pt'].sum() if prev_game['p2_net_pt'].sum() != 0 else 0
    # now_x_ls.append(net_rate)
    # # x12
    # count_ones = (now_game['serve_no'] == 1).sum()
    # # 计算比例
    # total_count = len(now_game)
    # proportion = count_ones / total_count
    # now_x_ls.append(proportion if now_point['server'].values[0] == 2 else 0)
    # label
    labels.append(1 if now_point['point_victor'].values[0] == 2 else 0)
    # id
    match_ids.append(now_point['match_id'].values[0])
    for i in range(len(now_x_ls)):
        x_ls[i].append(now_x_ls[i])

dataset = pd.DataFrame({
    'set_win': x_ls[0],
    'ace': x_ls[1],
    'server': x_ls[2],
    'speed': x_ls[3],
    'run': x_ls[4],
    'rally_count': x_ls[5],
    'AD_score': x_ls[6],
    'continue_wins': x_ls[7],
    'unforced_error': x_ls[8],
    'break_rate': x_ls[9],
    'net_rate': x_ls[11],
    'serve_rate': x_ls[12],
    'label': labels,
    'match_id': match_ids,
})
dataset.to_excel('../data/woman_pca_origin_data_2.xlsx', index=False)
scaler = MinMaxScaler()
columns = dataset.columns[:-2]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_excel('../data/woman_pca_standard_data_2.xlsx', index=False)

