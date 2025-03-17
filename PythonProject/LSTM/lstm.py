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
from sklearn.metrics import mean_absolute_error

start_time = time.time()
# 导入数据
file_path = r'C:\Users\DELL\Desktop\data用于lstm.xls'
sheet_a = '波浪'
sheet_b = '纵荡'
response = pd.read_excel(file_path, sheet_name=sheet_b)

output = "LSTM--Heave--Units128--batch32--input100s--Adam结果"
if not os.path.exists(output):
    os.makedirs(output)
#整理数据
response_numpy = response.to_numpy()
X_response = response_numpy[:, :]
X_response = X_response.reshape(X_response.shape[0], 1, X_response.shape[1])
print('X_response shape:', X_response.shape)
scaler_response = MinMaxScaler()
X_response = scaler_response.fit_transform(X_response.reshape(-1, 1)).reshape(X_response.shape)

# 定义滑动窗口函数（为离线数据）
# def create_dataset(motion_time_series, window_size):
#     X, y = [], []
#     for i in range(len(motion_time_series) - window_size):
#         window = motion_time_series[i:i + window_size]  # 形状 (window_size, n_features)
#         X.append(window)
#         y.append(motion_time_series[i + window_size]) #y是一个数（11）
#     return np.array(X), np.array(y)

# 定义滑动窗口函数
def create_dataset(motion_time_series, window_size):
    X, y = [], []
    # 添加窗口状态缓存
    window_buffer = []
    for i in range(len(motion_time_series)):
        if len(window_buffer) < window_size:
            window_buffer.append(motion_time_series[i])
        else:
            # 实时更新：移除最早数据点，添加最新数据点
            X.append(np.array(window_buffer))  # 使用完整窗口
            y.append(motion_time_series[i])    # 预测下一个点
            window_buffer.pop(0)
            window_buffer.append(motion_time_series[i])
    return np.array(X), np.array(y)
# 新增实时预测函数
class RealTimePredictor:
    def __init__(self, model, window_size, scaler):
        self.model = model
        self.window_size = window_size
        self.scaler = scaler
        self.window_buffer = []

    def update(self, new_data):
        # 数据标准化
        new_data = self.scaler.transform(new_data.reshape(-1, 1)).flatten()
        # 维护实时窗口
        if len(self.window_buffer) >= self.window_size:
            self.window_buffer.pop(0)
        self.window_buffer.append(new_data)
        # 生成预测输入
        if len(self.window_buffer) == self.window_size:
            input_data = np.array(self.window_buffer).reshape(1, self.window_size, -1)
            prediction = self.model.predict(input_data)
            return self.scaler.inverse_transform(prediction)
        return None

#递减学习率
def lr_schedule(epoch, lr):
    if epoch < 5:
        lr=0.001
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

#滑动窗口训练
def sliding_window_lstm_train(motion_time_series, window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, X_response.shape[1])),  # 添加Input层
        LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(64, activation='tanh', kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    #调用前面生成的x和y
    X, y = create_dataset(motion_time_series, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    history = model.fit(
        X_train, y_train,
        epochs=50,  # 增加训练轮数以确保充分收敛
        batch_size=64,  # 减小batch size以提高训练稳定性
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
    )
    # # 在训练集上评估模型
    # train_loss = model.evaluate(X_train, y_train)
    # print(f"Train Loss: {train_loss}")
    # # 在测试集上评估模型
    # test_loss = model.evaluate(X_test, y_test)
    # print(f"Test Loss: {test_loss}")
    # # 计算均方根误差
    # y_train_pred = model.predict(X_train)
    # train_rmse = root_mean_squared_error(y_train.reshape(-1), y_train_pred.reshape(-1))
    # print(f"Train RMSE: {train_rmse}")
    # y_test_pred = model.predict(X_test)
    # test_rmse = root_mean_squared_error(y_test.reshape(-1), y_test_pred.reshape(-1))
    # print(f"Test RMSE: {test_rmse}")

    # 添加预测和返回值
    predictions = model.predict(X_test)
    return model,predictions, history, y_train, y_test, scaler_response

motion_time_series = X_response
print('motion_time_series', X_response.shape)
window_size =50
#调用函数（重点代码）
model,predictions, history, y_train, y_test, scaler_response =sliding_window_lstm_train(motion_time_series, window_size)
# 初始化实时预测器
rt_predictor = RealTimePredictor(model, window_size, scaler_response)
predictions = predictions.reshape(-1, 1)
print('predictions', predictions.shape)
print('y_test', y_test.shape)
# 修改切片维度大小
y_train_first_dim = y_train[:predictions.shape[0], 0]  # 根据预测长度切片
y_test_first_dim = y_test[:predictions.shape[0], 0]

# 逆归一化函数
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

# 对预测结果调用逆归一化函数
predictions = inverse_transform(scaler_response, predictions)
y_train_first_dim = inverse_transform(scaler_response, y_train_first_dim.reshape(-1, 1)).reshape(-1)
y_test_first_dim = inverse_transform(scaler_response, y_test_first_dim.reshape(-1, 1)).reshape(-1)
# 将预测结果保存到Excel
results_df = pd.DataFrame({
    'y_test': y_test_first_dim,
    'predictions': predictions.flatten()
})
results_df.to_excel(os.path.join(output, 'predictions_results.xlsx'), index=False)

# 可视化预测结果（和训练集）
plt.figure(figsize=(10, 6))
plt.plot(y_train_first_dim, label='y_train ', color='blue', linestyle='-', linewidth=2)
plt.plot(predictions, label='predictions', color='red', linestyle='--', linewidth=1)
plt.title('y_train_first_dim  vs predictions')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output, 'y_train_first_dim_t_vs_predictions.png'))
plt.close()
# 可视化预测结果（和测试集）
plt.figure(figsize=(10, 6))
plt.plot(y_test_first_dim, label='y_test', color='blue', linestyle='-', linewidth=2)
plt.plot(predictions, label='predictions', color='red', linestyle='--', linewidth=1)
plt.title('y_test_first_dim vs predictions')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output, 'y_test_first_dim_t_vs_predictions.png'))
plt.close()

# 可视化训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'Training_History.png'))
    plt.close()
plot_training_history(history)
# 修复模型保存部分（取消注释并修改）
model_json = model.to_json()
with open(os.path.join(output, "model.json"), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(output, "model.weights.h5"))


