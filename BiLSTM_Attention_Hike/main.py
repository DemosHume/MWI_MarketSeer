# main.py
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from BiLSTM_Attention_Hike.data import get_mark_dada_df
from model import DualHeadBiLSTM
from predict import predict_top_rising, load_model
from train import train_model

# === 配置运行模式: 'train' | 'continue' | 'predict' ===
MODE = 'train'  # <-- 修改这里控制运行模式



df = get_mark_dada_df()
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

    model = train_model(x_data, y_data,
                        input_size=input_size,
                        n_future=n_future,
                        device=device,
                        model=model,
                        epoch=300,
                        lr=0.001)

elif MODE == 'predict':
    model = load_model(input_size=input_size, output_size=n_future, device=device)
    top_items = predict_top_rising(df.iloc[:, 1:], model, scaler, n_past, device, top_k=5)
    print("Top predicted items likely to rise in price:")
    for name, score in top_items:
        print(f"{name}: {score:.2%}")
