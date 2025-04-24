import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 如果目录下不存在 market.db 文件，从上级目录复制一个过来
import os
import shutil

from tqdm import tqdm

from utils import timestamp_to_beijing_time
from utils import query

if not os.path.exists('market.db'):
    shutil.copyfile('../market.db', 'market.db')

import sqlite3
import pandas as pd
from datetime import datetime
import pytz



latest_time = query('SELECT time FROM ask ORDER BY time DESC LIMIT 1')['time'].iloc[0]
print(f"最新数据\t北京时间: {timestamp_to_beijing_time(latest_time)}")


import numpy as np

ask_data = query('SELECT * FROM ask')
ask_data.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
ask_data = ask_data.interpolate(method='linear', axis=0, limit_direction='both')
ask_data = ask_data.dropna(axis=1, how='all')

# 过滤异常值
quantile_90 = ask_data.quantile(0.9)
ask_data = ask_data.apply(lambda x: np.where(x > 1.5 * quantile_90[x.name], np.nan, x))
ask_data = pd.DataFrame(ask_data, columns=ask_data.columns)
ask_data = ask_data.interpolate(method='linear', axis=0, limit_direction='both')

# 删除最大值大于100M的列（保留 time 列）
max_values = ask_data.max()
ask_data = ask_data.loc[:, (max_values < 1e6) | (ask_data.columns == 'time')]

# 删除完全相同的列
ask_data = ask_data.loc[:, ~ask_data.T.duplicated()]




def create_labels(predicted_prices, threshold=0.05):
    # 只看最后一个时间步的数据（也就是未来第n_future小时）
    final_prices = predicted_prices[-1]
    price_changes = (final_prices - predicted_prices[0]) / predicted_prices[0]
    labels = (price_changes > threshold).astype(int)
    return labels  # 返回形状为 [商品数]


def create_sequences(data, n_past, n_future, threshold=0.05):
    x, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        x.append(data[i:i + n_past])
        # y 只基于最后一个未来时间点
        y.append(create_labels(data[i + n_past:i + n_past + n_future], threshold))
    return np.array(x), np.array(y)

# 数据准备
df = ask_data.copy()
n_past, n_future = 72, 2

data = df.iloc[:, 1:].values

# 每列最大值（加上一个小数防止除以0）
col_max = data.max(axis=0)
col_max[col_max == 0] = 1  # 避免除以 0
# 每个值除以对应列的最大值
data = data / col_max

# 归一化，每个值除以自己列的最大值
if np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.any(data == -1):
    print("Data contains NaN, Inf, or -1!")


x_data, y_data = create_sequences(data, n_past, n_future)

# 定义数据集类
class PriceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# 定义BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.sigmoid(self.fc(out))  # Apply sigmoid for binary classification

# 初始化模型、损失函数和优化器
input_size = x_data.shape[2]
output_size = y_data.shape[1]  # Output size is 1 for each future item (whether it will rise by 5% or not)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BiLSTMModel(input_size, 256, output_size).to(device)
train_loader = DataLoader(PriceDataset(x_data, y_data), batch_size=64, shuffle=True)

# 损失函数
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epoch_num = 300
for epoch in range(epoch_num):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch).view(y_batch.size(0), -1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{epoch_num}] Loss: {total_loss / len(train_loader):.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 评估模型
model.eval()
data = df.iloc[:, 1:].values[-n_past:]
data = data / col_max  # 使用相同的归一化方法
input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    predicted_probabilities = model(input_tensor).view(-1).cpu().numpy()

# 输出涨价预测
top_items_columns = df.columns[1:]
for idx, prob in enumerate(predicted_probabilities):
    if prob > 0.5:
        print(f"{top_items_columns[idx]} 预测将在未来两小时内涨价超过 5%")
