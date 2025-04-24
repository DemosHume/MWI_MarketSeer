# predict.py
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import DualHeadBiLSTM
from utils import query


def load_model(input_size, output_size, device):
    model = DualHeadBiLSTM(input_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    return model


def predict_top_rising(df, model, scaler, n_past, device, top_k=5):
    # 取最新一段数据用于预测
    recent_data = df.iloc[-n_past:].values
    recent_scaled = scaler.transform(recent_data)
    x = torch.tensor(recent_scaled[np.newaxis, :, :], dtype=torch.float32).to(device)

    with torch.no_grad():
        _, cls_out = model(x)
        cls_out = cls_out.squeeze(0).cpu().numpy()  # shape: (input_size,)

    top_indices = cls_out.argsort()[::-1][:top_k]
    top_scores = cls_out[top_indices]
    top_items = df.columns[top_indices]

    return list(zip(top_items, top_scores))



