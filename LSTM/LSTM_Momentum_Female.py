import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.losses import mean_squared_error
from keras.src.saving.saving_api import load_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Lambda
import tensorflow as tf

Player = 1
# 读取 Excel 文件
dataset = pd.read_excel(f'../data/woman_momentum_{Player}.xlsx')
dataset = dataset[dataset['match_id'] != '2023-wimbledon-1701']
columns = dataset.columns[:-5]


# 读取特征数据
features = dataset[columns].values
num_augmented_samples = 2
# 扰动特征数据
augmented_features = []
for _ in range(num_augmented_samples):
    augmented_feature = features.copy()  # 创建特征数据的副本
    augmented_feature += np.random.normal(loc=0, scale=0.1, size=augmented_feature.shape)  # 添加高斯噪声
    augmented_features.append(augmented_feature)
augmented_features = np.concatenate(augmented_features)
# 将增强的特征数据与原始数据合并
augmented_features = np.concatenate((features, augmented_features), axis=0)

# 假设你的特征数据保存在 X 中，标签数据保存在 y 中
X_train, X_test, y_train, y_test = train_test_split(dataset[columns].values, dataset['sign_reversal'].values,
                                                    test_size=0.2,
                                                    random_state=42)
f_len = X_train.shape[1]
# 调整数据形状为适合 LSTM 模型
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape[0], X_train.shape[1])
Train = False
# 定义廉价丢弃层的比例
dropout_rate = 0.2

if Train:
    # 构建LSTM模型
    model = Sequential()

    # 添加廉价丢弃层
    model.add(LSTM(64, input_shape=(1, f_len)))  # 假设每个时间步的特征维度为input_dim
    model.add(Lambda(lambda x: x * (1 - dropout_rate)))
    model.add(Dense(1, activation='relu'))  # 二分类问题的输出层，可以根据需要进行修改
    learning_rate = 0.001

    # 定义优化器，并设置初始学习率
    optimizer = Adam(learning_rate=learning_rate)

    # 在训练过程中使用ReduceLROnPlateau回调来动态调整学习率
    lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.001)
    # 编译模型时指定优化器和回调
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32,
                        callbacks=[lr_scheduler])

    # 绘制图像
    plt.figure(figsize=(12, 4))

    # 绘制训练集和验证集的损失
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # 绘制训练集和验证集的损失
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    model.save(f"../models/woman_simple_model_{Player}_momentum_reversal.keras")


else:
    df = pd.read_excel('../data/woman_momentum_1.xlsx')
    index = df[df.match_id == '2023-wimbledon-2701'].reset_index(drop=True).index
    df = df.iloc[index]
    # 加载已保存的模型
    model = load_model('../models/woman_simple_model_1_momentum_reversal.keras', compile=False, safe_mode=False)
    # 准备测试数据
    X_Test = np.reshape(df[columns], (df[columns].shape[0], 1, f_len))
    Y_Test = np.array(df['sign_reversal'])
    # df2 = pd.read_excel('../data/momentum_2.xlsx')
    # index = df2[df2.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
    # df2 = df2.iloc[index]
    # # 加载已保存的模型
    # model = load_model('../models/simple_model_2_momentum_reversal.keras', compile=False, safe_mode=False)
    # y_pred2 = model.predict(df2[columns])

    # 将预测结果和实际结果转换为标量值（如果是one-hot编码，可使用np.argmax转换）
    # 评估模型
    # 使用LSTM模型对测试数据进行预测
    predictions = model.predict(X_Test)
    predictions = np.round(predictions).flatten()  # 将概率值转换为二分类标签（0或1）
    # 将预测结果和实际结果转换为标量值（如果是one-hot编码，可使用np.argmax转换）
    predictions = np.squeeze(predictions)
    actual_values = np.squeeze(Y_Test)

    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    df = pd.read_csv(r"../data/2023-wimbledon-female-matches.csv")
    df.loc[(df.p1_score == 'AD', 'p1_score')] = 50
    df.loc[(df.p2_score == 'AD', 'p2_score')] = 50
    df['p1_score'] = df['p1_score'].astype(int)
    df['p2_score'] = df['p2_score'].astype(int)
    # 删除缺省值
    df.dropna(subset=['speed_mph'], inplace=True)
    df.drop(df[(df['point_no'] == '0X') | (df['point_no'] == '0Y')].index, inplace=True)
    df['point_no'] = df['point_no'].astype(int)
    df = df[df.match_id == '2023-wimbledon-2701']
    print(len(df['point_no']), len(predictions))
    # 绘制预测结果曲线
    plt.scatter(x=df['point_no'], y=predictions, label='Predictions')

    # 绘制实际结果曲线
    plt.scatter(x=df['point_no'], y=actual_values, label='Actual')

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('LSTM Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # 显示图形
    plt.show()

    # 计算准确率
    accuracy = np.mean(predictions == Y_Test)
    print('Accuracy:', accuracy)
