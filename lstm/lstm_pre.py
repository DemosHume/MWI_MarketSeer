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

from sklearn.preprocessing import StandardScaler

df = ask_data.copy()
n_past, n_future = 72, 2

data = df.iloc[:, 1:].values

if np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.any(data == -1):
    print("Data contains NaN, Inf, or -1!")

scaler = StandardScaler()
data = scaler.fit_transform(data)

def create_sequences(data, n_past, n_future):
    x, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        x.append(data[i:i+n_past])
        y.append(data[i+n_past:i+n_past+n_future])
    return np.array(x), np.array(y)

x_data, y_data = create_sequences(data, n_past, n_future)



import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class PriceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.fc(out)

input_size = x_data.shape[2]
output_size = n_future * input_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = BiLSTMModel(input_size, 256, output_size).to(device)
train_loader = DataLoader(PriceDataset(x_data, y_data), batch_size=64, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epoch_num = 300
for epoch in tqdm(range(epoch_num)):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch.view(y_batch.size(0), -1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epoch_num}] Loss: {total_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'model.pth')


model.eval()
data = df.iloc[:, 1:].values[-n_past:]
data = scaler.transform(data)
input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    predicted_prices = model(input_tensor).view(n_future, data.shape[-1]).detach().cpu().numpy()

top_10_items_columns = df.columns[1:]
price_changes = [(predicted_prices[:, i] - predicted_prices[0, i]) / predicted_prices[0, i] for i in range(input_size)]
final_price_changes = [float(change[-1]) for change in price_changes]

sorted_indices = sorted(range(len(final_price_changes)), key=lambda i: final_price_changes[i], reverse=True)
top_10_items = sorted_indices[:5]

print(f"最新数据\t北京时间: {timestamp_to_beijing_time(latest_time)}")

for idx in top_10_items:
    print(f"{idx} 商品\t {top_10_items_columns[idx]} \t的涨幅为 {final_price_changes[idx]:.2%}")


top_item_cols = top_10_items_columns[top_10_items]
columns_to_display = ['time'] + top_item_cols.tolist()
last_24_hours_data = df.loc[df.index[-24:], columns_to_display]

# 归一化处理
for item in top_item_cols:
    last_24_hours_data[item] /= last_24_hours_data[item].iloc[0]

plt.figure(figsize=(12, 6))
for item in top_item_cols:
    plt.plot(last_24_hours_data['time'], last_24_hours_data[item], label=item)

plt.title('Top 5 Item Price Trends in Last 24 Hours (Normalized)')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend(title="Items", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('last_24_hours_trends.png')
plt.close()

