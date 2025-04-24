# data.py
import os
import shutil

import numpy as np
import torch

from utils import query, timestamp_to_beijing_time


class PriceDataset(torch.utils.data.Dataset):
    def __init__(self, x, y,device='cpu'):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.cls = (self.y[:, -1, :] - self.x[:, -1, :]) / self.x[:, -1, :] > 0.02
        self.cls = self.cls.float()

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].view(-1), self.cls[idx]


def get_mark_dada_df():
    # 复制数据库（仅首次）
    if not os.path.exists('market.db'):
        shutil.copy('../../market.db', 'market.db')
    latest_time = query('SELECT time FROM ask ORDER BY time DESC LIMIT 1')['time'].iloc[0]
    print(f"{latest_time}\n最新数据\t北京时间: {timestamp_to_beijing_time(latest_time)}")

    # 数据准备
    df = query('SELECT * FROM ask')
    df.replace([np.inf, -np.inf, -1], np.nan, inplace=True)

    # 异常值剔除（IQR方法）
    q_value = 1.5
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('time', errors='ignore')  # 数值列排除 time
    for col in numeric_cols:
        col_Q1 = df[col].quantile(0.25)
        col_Q3 = df[col].quantile(0.75)
        col_IQR = col_Q3 - col_Q1
        lower_bound = col_Q1 - q_value * col_IQR
        upper_bound = col_Q3 + q_value * col_IQR
        # 将异常值替换为np.nan
        df[col] = df[col].mask((df[col] < lower_bound) | (df[col] > upper_bound), np.nan)
    # 线性插值填充
    df = df.interpolate(method='linear', axis=0, limit_direction='both')


    # 清洗：去掉过大值列、重复列、全NaN列
    df = df.loc[:, (df.max() < 1e6) | (df.columns == 'time')]
    df = df.loc[:, ~df.T.duplicated()]
    df = df.dropna(axis=1, how='all')
    return df


if __name__ == '__main__':
    # 测试数据加载
    df = get_mark_dada_df()
    print(df.shape)
    print(df.head())