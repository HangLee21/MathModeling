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
for i in range(15):
    x_ls.append([])

for match_id, set_no, game_no, point_no in zip(df.match_id, df.set_no, df.game_no, df.point_no):
    now_match = df[df.match_id == match_id]
    now_set = now_match[now_match.set_no == set_no]
    now_game = now_set[now_set.game_no == game_no]
    now_point = now_game[now_game.point_no == point_no]
    last_point = now_game[now_game.point_no == point_no - 1]
    # x0 当前set 赢了几局
    # x1 当前game 领先几分
    # x2 是否为server
    # x3 上一point 是否得分
    # x4 本match的领先程度
    # x5 本game是否发球得分
    # x6 本game是否回击得分
    # x7 本game是否出现double fault
    # x8 是否存在unforced error
    # x9 上网次数与上网得分比例
    # x10 本盘对方发球获得得分机会和实际获得得分的比例
    # x11 本match内总计跑图里程
    # x12 最近三个point跑图里程
    # x13 上个point跑图里程
    # x14 发球配速
    now_x_ls = []
    # x0
    now_x_ls.append(now_point['p1_games'].values[0])
    # x1
    now_x_ls.append(now_point['p1_score'].values[0] - now_point['p2_score'].values[0])
    # x2
    now_x_ls.append(1 if now_point['serve_no'].values[0] == 1 else 0)
    #  x3
    if len(last_point['p1_score']) > 0:
        now_x_ls.append(0 if now_point['p1_score'].values[0] == last_point['p1_score'].values[0] else 1)
    else:
        now_x_ls.append(0)
    # x4
    now_x_ls.append(now_point['p1_sets'].values[0] - now_point['p2_sets'].values[0])
    # x5
    now_x_ls.append(1 if 1 in now_game['p1_ace'].values else 0)
    # x6
    now_x_ls.append(1 if 1 in now_game['p1_winner'].values else 0)
    # x7
    now_x_ls.append(1 if 1 in now_game['p1_double_fault'].values else 0)
    # x8
    now_x_ls.append(1 if 1 in now_game['p1_unf_err'].values else 0)
    # x9
    now_x_ls.append(now_game['p1_net_pt_won'].sum()/ now_game['p1_net_pt'].sum() if now_game['p1_net_pt'].sum() != 0 else 0)
    # x10
    now_x_ls.append(now_set['p1_break_pt_won'].sum()/ now_set['p1_break_pt'].sum() if now_set['p1_break_pt'].sum() != 0 else 0)
    # x11
    index = now_match.index.tolist().index(now_point.index.tolist()[0])
    now_x_ls.append(now_match.iloc[:index+1]['p1_distance_run'].sum())
    # x12
    now_x_ls.append(now_match.iloc[index - 2:index + 1]['p1_distance_run'].sum())
    # x13
    now_x_ls.append(now_point['p1_distance_run'].values[0])
    # x14
    now_x_ls.append(now_point['speed_mph'].values[0] if now_point['server'].values[0] == 1 else 0)
    # label
    labels.append(1 if now_point['point_victor'].values == 1 else 0)
    for i in range(len(now_x_ls)):
        x_ls[i].append(now_x_ls[i])

dataset = pd.DataFrame({
    'x0': x_ls[0],
    'x1': x_ls[1],
    'x2': x_ls[2],
    'x3': x_ls[3],
    'x4': x_ls[4],
    'x5': x_ls[5],
    'x6': x_ls[6],
    'x7': x_ls[7],
    'x8': x_ls[8],
    'x9': x_ls[9],
    'x10': x_ls[10],
    'x11': x_ls[11],
    'x12': x_ls[12],
    'x13': x_ls[13],
    'x14': x_ls[14],
    'label': labels,
})

dataset.to_excel('../data/origin_data.xlsx', index=False)
scaler = MinMaxScaler()
columns = dataset.columns[:-1]
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_excel('../data/standard_data.xlsx', index=False)
