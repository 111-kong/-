import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出进行预测
        return out


# 滑动窗口函数
def sliding_window(motion_time_series, window_size):
    X_response_temp = []
    y_temp = []
    for i in range(len(motion_time_series) - window_size):
        X_response_temp.append(motion_time_series[i:i + window_size].reshape(window_size, 1))
        y_temp.append(motion_time_series[i + window_size].reshape(1, 1))
    return np.array(X_response_temp), np.array(y_temp)


def load_and_preprocess_data(file_path, sheet_a, sheet_b):
    wave = pd.read_excel(file_path, sheet_name=sheet_a)
    response = pd.read_excel(file_path, sheet_name=sheet_b)
    alltime = int(10800)
    ds = 0.3
    num_datapoint = int(alltime / ds)
    wave_data = []
    for i in range(len(wave.columns)):
        wave_data.append(wave[wave.columns[i]].values)
    X_wave = np.array(wave_data).reshape(num_datapoint, 1, 1)
    response_numpy = response.to_numpy()
    X_response = response_numpy[:, :]
    scaler_wave = MinMaxScaler()
    X_wave = scaler_wave.fit_transform(X_wave.reshape(-1, 1)).reshape(X_wave.shape)
    scaler_response = MinMaxScaler()
    X_response = scaler_response.fit_transform(X_response.reshape(-1, 1)).reshape(X_response.shape)
    return X_wave, X_response


def train_model(model, X_train, y_train, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        torch.cuda.empty_cache()  # 释放 GPU 缓存
        del outputs  # 删除不再使用的变量


def test_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
        return predictions


def main():
    start_time = time.time()
    file_path = r'C:\Users\DELL\Desktop\data1.xls'
    sheet_a = '波浪'
    sheet_b = '纵荡'
    output = "LSTM--Heave--Units128--batch32--input100s--Adam结果"
    if not os.path.exists(output):
        os.makedirs(output)
    X_wave, X_response = load_and_preprocess_data(file_path, sheet_a, sheet_b)
    print('X_wave shape:', X_wave.shape)
    print('X_response shape:', X_response.shape)
    wave_surface = X_wave
    motion_time_series = X_response
    window_size = 150
    num_steps = 4000
    X_response, y = sliding_window(motion_time_series, window_size)
    X = torch.from_numpy(X_response).float().requires_grad_(False)  # 避免存储梯度
    y = torch.from_numpy(y).float().requires_grad_(False)
    print('X', X.shape)
    print('y', y.shape)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    input_size = 1
    hidden_size = 30
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(torch.float16)  # 使用半精度浮点数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 20
    train_model(model, X_train, y_train, num_epochs, criterion, optimizer)
    predictions = test_model(model, X_test, y_test, criterion)
    plt.plot(y_test.numpy(), label='True')
    plt.plot(predictions.numpy(), label='Predicted')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()