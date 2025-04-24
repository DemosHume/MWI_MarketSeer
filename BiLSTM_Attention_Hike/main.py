# main.py
import os, shutil
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from train import train_model
from model import DualHeadBiLSTM
from utils import query, timestamp_to_beijing_time
from predict import predict_top_rising, load_model

# === 配置运行模式: 'train' | 'continue' | 'predict' ===
MODE = 'predict'  # <-- 修改这里控制运行模式

# 复制数据库（仅首次）
if not os.path.exists('market.db'):
    shutil.copy('../../market.db', 'market.db')
latest_time = query('SELECT time FROM ask ORDER BY time DESC LIMIT 1')['time'].iloc[0]
print(f"{latest_time}\n最新数据\t北京时间: {timestamp_to_beijing_time(latest_time)}")

# 数据准备
df = query('SELECT * FROM ask')
df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
df = df.interpolate(method='linear', axis=0, limit_direction='both').dropna(axis=1, how='all')
df = df.loc[:, (df.max() < 1e6) | (df.columns == 'time')]
df = df.loc[:, ~df.T.duplicated()]
columns = df.columns[1:]  # 去掉 time 列
data = df.iloc[:, 1:].values  # 去掉 time 列的数据部分

n_past, n_future = 72, 2
scaler = StandardScaler()
data = scaler.fit_transform(data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'None'}")

def create_sequences(data, n_past, n_future):
    x, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        x.append(data[i:i+n_past])
        y.append(data[i+n_past:i+n_past+n_future])
    return np.array(x), np.array(y)

x_data, y_data = create_sequences(data, n_past, n_future)
input_size = x_data.shape[2]

if MODE in ['train', 'continue']:
    model = None
    if MODE == 'continue' and os.path.exists('model.pth'):
        model = DualHeadBiLSTM(input_size, output_size=n_future).to(device)
        model.load_state_dict(torch.load('model.pth'))
        print("Loaded existing model for continued training.")
    else:
        print("Starting fresh training.")

    model = train_model(x_data, y_data, input_size=input_size, n_future=n_future, device=device, model=model)

elif MODE == 'predict':
    model = load_model(input_size=input_size, output_size=n_future, device=device)
    top_items = predict_top_rising(df.iloc[:, 1:], model, scaler, n_past, device, top_k=5)
    print("Top predicted items likely to rise in price:")
    for name, score in top_items:
        print(f"{name}: {score:.2%}")
