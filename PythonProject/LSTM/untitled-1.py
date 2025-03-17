from keras import Model
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.layers import Add

start_time = time.time()

file_path = r'C:\Users\DELL\Desktop\data1.xls'
sheet_a = '波浪'
sheet_b = '纵荡'
wave = pd.read_excel(file_path, sheet_name=sheet_a)
response = pd.read_excel(file_path, sheet_name=sheet_b)

output = "LSTM--Heave--Units128--batch32--input100s--Adam结果"
if not os.path.exists(output):
    os.makedirs(output)

alltime = int(30503)
ds = 1
response_numpy = response.to_numpy()
X_response = response_numpy[:, :]
X_response = X_response.reshape(X_response.shape[0], 1, X_response.shape[1])


print('X_response shape:', X_response.shape)


scaler_response = MinMaxScaler()
X_response = scaler_response.fit_transform(X_response.reshape(-1, 1)).reshape(X_response.shape)

# 实时更新的滑动窗口函数
def create_dataset(motion_time_series, window_size):
    X, y = [], []
    for i in range(len(motion_time_series) - window_size):
        X.append(motion_time_series[i:i+window_size])
        y.append(motion_time_series[i+window_size])
    return np.array(X), np.array(y)

# 递减学习率
def lr_schedule(epoch, lr):
    return lr * (0.5 ** (epoch // 100))

def sliding_window_lstm_predicts(motion_time_series, window_size, num_steps):
    model = tf.keras.Sequential([
        LSTM(128, return_sequences=True, input_shape=(window_size, 1), kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(32, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    
    X, y = create_dataset(motion_time_series, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    history = model.fit(X_train, y_train, epochs=35, batch_size=128, validation_data=(X_test, y_test),
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])
    
    predictions = model.predict(X_test)
    
    return predictions, history, y_train, y_test

motion_time_series = X_response

print('motion_time_series', X_response.shape)
window_size = 50
num_steps = 1000

predictions, history, y_train, y_test = sliding_window_lstm_predicts(motion_time_series, window_size, num_steps)

predictions = predictions.reshape(1000, 1)
print('predictions', predictions.shape)
print('y_test', y_test.shape)
y_train_first_dim = y_train[:1000, 0, 0]
y_test_first_dim = y_test[:1000, 0, 0]

# 逆归一化操作
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

# 对预测结果进行逆归一化
predictions = inverse_transform(scaler_response, predictions)
y_train_first_dim = inverse_transform(scaler_response, y_train_first_dim.reshape(-1, 1)).reshape(-1)
y_test_first_dim = inverse_transform(scaler_response, y_test_first_dim.reshape(-1, 1)).reshape(-1)

plt.figure(figsize=(10, 6))
plt.plot(y_train_first_dim, label='y_train ', color='blue', linestyle='-', linewidth=2)
plt.plot(predictions, label='predictions', color='red', linestyle='--', linewidth=1)
plt.title('y_train_first_dim  vs predictions')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output, 'y_train_first_dim_t_vs_predictions.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(y_test_first_dim, label='y_test ', color='blue', linestyle='-', linewidth=2)
plt.plot(predictions, label='predictions', color='red', linestyle='--', linewidth=1)
plt.title('y_test_first_dim  vs predictions')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output, 'y_test_first_dim_t_vs_predictions.png'))
plt.close()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output, 'Loss_vs_Epochs.png'))
plt.close()

'''
plt.figure(figsize=(10, 6))
plt.bar(['Train', 'Test'], [rmse_train, rmse_test], color=['blue', 'red'])
plt.title('RMSE for Train and Test Sets')
plt.xlabel('Dataset')
plt.ylabel('RMSE')
plt.show()
'''
df_predictions = pd.DataFrame(predictions, columns=['Predictions'])
df_predictions.to_excel(os.path.join(output, 'Predictions.xlsx'), index=False)
df_y_train_first_dim = pd.DataFrame(y_train_first_dim, columns=['Train'])
df_y_train_first_dim.to_excel(os.path.join(output, 'y_train_first_dim.xlsx'), index=False)
