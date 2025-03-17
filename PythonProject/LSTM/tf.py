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
num_datapoint = int(alltime / ds)

response_numpy = response.to_numpy()
X_response = response_numpy[:, :]
X_response = X_response.reshape(X_response.shape[0], 1, X_response.shape[1])


print('X_response shape:', X_response.shape)


scaler_response = MinMaxScaler()
X_response = scaler_response.fit_transform(X_response.reshape(-1, 1)).reshape(X_response.shape)


## 滑动窗口函数
def create_dataset(motion_time_series, window_size):
    X_response_temp = []
    y_temp = []
    for i in range(len(motion_time_series) - window_size):
        X_response_temp.append(motion_time_series[i:i + window_size].reshape(window_size, 1, 1))
        y_temp.append(motion_time_series[i + window_size].reshape(1, 1))


    return tf.convert_to_tensor(np.array(X_response_temp)), tf.convert_to_tensor(np.array(y_temp))


## 递减学习率
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        lr = 0.01
        return lr * 0.1
    elif epoch < 30:
        lr = 0.001
        return lr * 0.1
    elif epoch < 40:
        lr = 0.0001
        return lr * 0.1
    elif epoch < 50:
        lr = 0.00001
        return lr * 0.1
    else:
        return lr * 0.01


def sliding_window_lstm_predicts(motion_time_series, window_size, num_steps):
    all_predictions = []
    train_data2 = motion_time_series[:int(0.8 * len(motion_time_series))]
    test_data2 = motion_time_series[int(0.8 * len(motion_time_series)):]
    X2_train, y_train = create_dataset(train_data2, window_size)
    X2_test, y_test = create_dataset(test_data2, window_size)


    X2_train = tf.reshape(X2_train, (X2_train.shape[0], window_size, 1))
    X2_test = tf.reshape(X2_test, (X2_test.shape[0], window_size, 1))


    print('X2_train.shape', X2_train.shape)
    print('X2_test.shape', X2_test.shape)


    # 输入层
    input2 = Input(shape=(window_size, 1))
    lstm2 = LSTM(units=150, activation='tanh')(input2)


    # 输出层
    output1 = Dense(units=80, activation='relu')(lstm2)
    output2 = Dropout(0.3)(output1)
    output2_1 = Dense(units=110, activation='relu')(output2)
    output2_2 = Dense(units=110, activation='relu')(output2_1)
    output3 = Dense(units=90, activation='relu')(output2_2)
    output4 = Dense(units=100)(output3)


    model = Model(inputs=input2, outputs=output3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # 自定义 RMSE 损失函数
    #def rmse_loss(y_true, y_pred):
      # return tf.sqrt((tf.square(y_true - y_pred)))
    #rmse = np.sqrt(mean_squared_error(X2_train, y_train))
    #mse = mean_squared_error(y_test, y_pred)
    # Bug 修复：将 X2_train 和 y_train 转换为 NumPy 数组
    from sklearn import metrics
# Bug 修复：将 X2_train_np 和 y_train_np 转换为 2 维数组
# Bug 修复：将 X2_train 和 y_train 转换为 NumPy 数组
    '''

    X2_train_np = X2_train.numpy()
    y_train_np = y_train.numpy()

    X2_train_np_2d = X2_train_np.reshape(X2_train_np.shape[0], -1)
    y_train_np_2d = y_train_np.reshape(y_train_np.shape[0], -1)
    '''
        # 自定义 RMSE 损失函数
    def rmse_loss(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    from tensorflow.keras.callbacks import EarlyStopping

    # 定义早退回调函数
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.007, patience=5, verbose=1)
    model.compile(optimizer=optimizer, loss=rmse_loss)

    # 添加学习率调度器
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    history = model.fit(X2_train, y_train, epochs=30, batch_size=256,
                      validation_data=(X2_test, y_test), verbose=1,callbacks=[lr_scheduler])


    predictions = []

    last_input2 = X2_test[-1]
    for _ in range(num_steps):
        next_prediction = model.predict(tf.reshape(last_input2, (1, window_size, 1)))[0][0]
        predictions.append(next_prediction)
        last_input2 = tf.concat([last_input2[1:], [[next_prediction]]], axis=0)


    return np.array(predictions), history, y_train, y_test, scaler_response

motion_time_series = X_response

print('motion_time_series', X_response.shape)
window_size = 50
num_steps = 1000

predictions, history, y_train, y_test, scaler_response = sliding_window_lstm_predicts(motion_time_series, window_size, num_steps)

predictions = predictions.reshape(1000, 1)
print('predictions', predictions.shape)
print('y_test', y_test.shape)
y_train_first_dim = y_train[:1000, 0, 0]
y_test_first_dim = y_test[:1000, 0, 0]


# 逆归一化操作
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)


# 对预测结果进行逆归一化
predictions = inverse_transform(scaler_response, predictions)# Bug 修复：将 y_train_first_dim 转换为 NumPy 数组
y_train_first_dim = inverse_transform(scaler_response, y_train_first_dim.numpy().reshape(-1, 1)).reshape(-1)
y_test_first_dim = inverse_transform(scaler_response, y_test_first_dim.numpy().reshape(-1, 1)).reshape(-1)



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