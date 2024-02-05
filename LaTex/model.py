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

Player = 2
dataset = pd.read_excel(f'../data/momentum_{Player}.xlsx')
dataset = dataset[dataset['match_id'] != '2023-wimbledon-1701']
columns = dataset.columns[:-5]
X_train, X_test, y_train, y_test = train_test_split(dataset[columns].values, dataset['sign_reversal'].values,
                                                    test_size=0.2,
                                                    random_state=42)
f_len = X_train.shape[1]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape[0], X_train.shape[1])
Train = False
dropout_rate = 0.2
model = Sequential()
model.add(LSTM(64, input_shape=(1, f_len)))
model.add(Lambda(lambda x: x * (1 - dropout_rate)))
model.add(Dense(1, activation='relu'))
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
lr_scheduler = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=32,
                    callbacks=[lr_scheduler, early_stopping])
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
model.save(f"../models/simple_model_{Player}_momentum_reversal.keras")



