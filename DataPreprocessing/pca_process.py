import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"../data/Wimbledon_featured_matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)

x_ls = []
labels = []
match_ids = []
for i in range(8):
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
    now_x_ls = []
    # x0
    now_x_ls.append(now_point['p1_games'].values[0] - now_point['p2_games'].values[0])
    # x1
    now_x_ls.append(1 if now_point['p1_ace'].values[0] == 1 else 0)
    # x2
    now_x_ls.append(1 if now_point['server'].values[0] == 1 else 0)
    # x3
    now_x_ls.append(now_point['speed_mph'].values[0] if now_point['serve_no'].values[0] == 1 else 0)
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
    # # x8
    # scores = 0
    # prev_game = now_set[now_set.game_no == game_no - 1]
    # if len(prev_game) != 0:

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
    'label': labels,
    'match_id': match_ids,
})
dataset.to_excel('../data/pca_origin_data_1.xlsx', index=False)
scaler = MinMaxScaler()
columns = dataset.columns[:-2]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_excel('../data/pca_standard_data_1.xlsx', index=False)


df = pd.read_csv(r"../data/Wimbledon_featured_matches.csv")
df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
# 删除缺省值
df.dropna(subset=['speed_mph'], inplace=True)

x_ls = []
labels = []
match_ids = []
for i in range(8):
    x_ls.append([])

for match_id, set_no, game_no, point_no in zip(df.match_id, df.set_no, df.game_no, df.point_no):
    now_match = df[df.match_id == match_id]
    now_set = now_match[now_match.set_no == set_no]
    now_game = now_set[now_set.game_no == game_no]
    now_point = now_game[now_game.point_no == point_no]
    last_point = now_game[now_game.point_no == point_no - 1]
    # x0 当前set领先局
    # x1 ace球
    # x2 是否发球
    # x3 球速
    # x4 跑动距离
    # x5 回击次数
    # x6 领先分数
    # x7 本game连续得分或失分局数
    now_x_ls = []
    # x0
    now_x_ls.append(now_point['p2_games'].values[0] - now_point['p1_games'].values[0])
    # x1
    now_x_ls.append(1 if now_point['p2_ace'].values[0] == 1 else 0)
    # x2
    now_x_ls.append(1 if now_point['server'].values[0] == 2 else 0)
    # x3
    now_x_ls.append(now_point['speed_mph'].values[0] if now_point['serve_no'].values[0] == 2 else 0)
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
    'label': labels,
    'match_id': match_ids,
})
dataset.to_excel('../data/pca_origin_data_2.xlsx', index=False)
scaler = MinMaxScaler()
columns = dataset.columns[:-2]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_excel('../data/pca_standard_data_2.xlsx', index=False)

