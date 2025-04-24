# main.py
import os, shutil
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from utils import query, timestamp_to_beijing_time
from train import train_model

# 拷贝数据库
if not os.path.exists('market.db'):
    shutil.copy('../../market.db', 'market.db')

df = query('SELECT * FROM ask')
df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
df = df.interpolate(method='linear', axis=0, limit_direction='both').dropna(axis=1, how='all')
df = df.loc[:, (df.max() < 1e6) | (df.columns == 'time')]
df = df.loc[:, ~df.T.duplicated()]

n_past, n_future = 72, 2
data = df.iloc[:, 1:].values
scaler = StandardScaler()
data = scaler.fit_transform(data)

def create_sequences(data, n_past, n_future):
    x, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        x.append(data[i:i+n_past])
        y.append(data[i+n_past:i+n_past+n_future])
    return np.array(x), np.array(y)

x_data, y_data = create_sequences(data, n_past, n_future)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(
    f"Using device: {device}, GPU: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'None'}"
)
model = train_model(x_data, y_data, input_size=x_data.shape[2], n_future=n_future, device=device)
