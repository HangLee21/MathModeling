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
dataset = pd.read_excel(f'../data/momentum_{Player}.xlsx')
dataset = dataset[dataset['match_id'] != '2023-wimbledon-1701']
columns = dataset.columns[:-3]

# 假设你的特征数据保存在 X 中，标签数据保存在 y 中
X_train, X_test, y_train, y_test = train_test_split(dataset[columns].values, dataset['Momentum'].values, test_size=0.2,
                                                    random_state=42)
# 调整数据形状为适合 LSTM 模型
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Train = False
# 定义廉价丢弃层的比例
dropout_rate = 0.2

if Train:
    # 构建LSTM模型
    model = Sequential()

    # 添加廉价丢弃层
    model.add(Lambda(lambda x: x * (1 - dropout_rate)))
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))  # 假设每个时间步的特征维度为input_dim
    model.add(Dense(1, activation='sigmoid'))  # 二分类问题的输出层，可以根据需要进行修改
    learning_rate = 0.001

    # 定义优化器，并设置初始学习率
    optimizer = Adam(learning_rate=learning_rate)

    # 在训练过程中使用ReduceLROnPlateau回调来动态调整学习率
    lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.001)
    # 编译模型时指定优化器和回调
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='min')
    # 训练模型
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32,
                        callbacks=[lr_scheduler, early_stopper])

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
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    model.save(f"../models/simple_model_{Player}.keras")


else:
    df = pd.read_excel('../data/momentum_1.xlsx')
    index = df[df.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
    df = df.iloc[index]
    # 加载已保存的模型
    model = load_model('../models/simple_model_1.keras', compile=False, safe_mode=False)
    y_pred = model.predict(df[columns])

    df2 = pd.read_excel('../data/momentum_2.xlsx')
    index = df2[df2.match_id == '2023-wimbledon-1701'].reset_index(drop=True).index
    df2 = df2.iloc[index]
    # 加载已保存的模型
    model = load_model('../models/simple_model_2.keras', compile=False, safe_mode=False)
    y_pred2 = model.predict(df2[columns])

    # 绘制预测结果
    plt.figure(figsize=(48, 6))
    plt.plot(df['Momentum'], label='True Player1')
    plt.plot(df2['Momentum'], label='True Player2')
    plt.plot(y_pred, label='Player1 Predicted')
    plt.plot(y_pred2, label='Player2 Predicted')
    plt.title('Prediction Results')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()



