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


def create_dataset(motion_time_series, window_size):
    X, y = [], []
    for i in range(len(motion_time_series) - window_size):
        X.append(motion_time_series[i:i + window_size])
        y.append(motion_time_series[i + window_size])
    return np.array(X), np.array(y)


# 递减学习率
def lr_schedule(epoch, lr):
    return lr * (0.5 ** (epoch // 100))


def sliding_window_lstm_train(motion_time_series, window_size, epochs=35, batch_size=128):
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

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])

    # 在训练集上评估模型
    train_loss = model.evaluate(X_train, y_train)
    print(f"Train Loss: {train_loss}")

    # 在测试集上评估模型
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}")

    # 计算均方根误差
    y_train_pred = model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train.reshape(-1), y_train_pred.reshape(-1))
    print(f"Train RMSE: {train_rmse}")

    y_test_pred = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test.reshape(-1), y_test_pred.reshape(-1))
    print(f"Test RMSE: {test_rmse}")

    # 计算平均绝对误差
    train_mae = mean_absolute_error(y_train.reshape(-1), y_train_pred.reshape(-1))
    print(f"Train MAE: {train_mae}")
    test_mae = mean_absolute_error(y_test.reshape(-1), y_test_pred.reshape(-1))
    print(f"Test MAE: {test_mae}")

    return model, history, X_test, y_test, y_test_pred


def plot_training_history(history, output):
    plt.figure(figsize=(12, 6))
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制学习率变化
    plt.subplot(1, 2, 2)
    lr = history.history.get('lr', [lr_schedule(i, 0.001) for i in range(len(history.history['loss']))])
    epochs = range(1, len(lr) + 1)
    plt.plot(epochs, lr, label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'Training_History.png'))
    plt.close()


def save_model(model, output):
    model_json = model.to_json()
    with open(os.path.join(output, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(output, "model.h5"))
    # 保存神经网络参数
    model.save(os.path.join(output, "full_model.h5"))


def load_and_continue_training(model_path, motion_time_series, window_size, epochs=35, batch_size=128):
    # 加载之前保存的模型结构和权重
    with open(os.path.join(model_path, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(model_path, "model.h5"))

    # 编译模型
    loaded_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

    # 创建新数据集
    X, y = create_dataset(motion_time_series, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 继续训练模型
    history = loaded_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                               callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])

    # 在训练集上评估模型
    train_loss = loaded_model.evaluate(X_train, y_train)
    print(f"Train Loss on new data: {train_loss}")

    # 在测试集上评估模型
    test_loss = loaded_model.evaluate(X_test, y_test)
    print(f"Test Loss on new data: {test_loss}")

    # 计算均方根误差
    y_train_pred = loaded_model.predict(X_train)
    train_rmse = root_mean_squared_error(y_train.reshape(-1), y_train_pred.reshape(-1))
    print(f"Train RMSE on new data: {train_rmse}")

    y_test_pred = loaded_model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test.reshape(-1), y_test_pred.reshape(-1))
    print(f"Test RMSE on new data: {test_rmse}")

    # 计算平均绝对误差
    train_mae = mean_absolute_error(y_train.reshape(-1), y_train_pred.reshape(-1))
    print(f"Train MAE on new data: {train_mae}")
    test_mae = mean_absolute_error(y_test.reshape(-1), y_test_pred.reshape(-1))
    print(f"Test MAE on new data: {test_mae}")

    return loaded_model, history, X_test, y_test, y_test_pred


def plot_test_set_predictions(y_test, y_test_pred, output):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reshape(-1), label='True Values', color='blue', linestyle='-', linewidth=2)
    plt.plot(y_test_pred.reshape(-1), label='Predicted Values', color='red', linestyle='--', linewidth=1)
    plt.title('Test Set Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(output, 'Test_Set_Predictions.png'))
    plt.close()


def main():
    window_size = 50
    epochs = 35
    batch_size = 128
    output = "LSTM--Heave--Units128--batch32--input100s--Adam结果"
    if not os.path.exists(output):
        os.makedirs(output)

    data_paths = [
        r'C:\Users\DELL\Desktop\data1.xls',
        r'C:\Users\DELL\Desktop\data2.xls',
        r'C:\Users\DELL\Desktop\data3.xls',
        r'C:\Users\DELL\Desktop\data4.xls',
        r'C:\Users\DELL\Desktop\data5.xls'
    ]

    # 初始化模型
    model = None
    history = None
    for i, data_path in enumerate(data_paths):
        print(f"Training on dataset {i + 1}")
        response = pd.read_excel(data_path)
        response_numpy = response.to_numpy()
        X_response = response_numpy[:, :]
        X_response = X_response.reshape(X_response.shape[0], 1, X_response.shape[1])

        scaler_response = MinMaxScaler()
        X_response = scaler_response.fit_transform(X_response.reshape(-1, 1)).reshape(X_response.shape)

        if model is None:
            model, history, X_test, y_test, y_test_pred = sliding_window_lstm_train(X_response, window_size, epochs,
                                                                                    batch_size)
        else:
            model, history, X_test, y_test, y_test_pred = load_and_continue_training(output, X_response, window_size,
                                                                                     epochs, batch_size)

        plot_training_history(history, output)
        plot_test_set_predictions(y_test, y_test_pred, output)
        save_model(model, output)


if __name__ == "__main__":
    main()