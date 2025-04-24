import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import DualHeadBiLSTM  # 你自己定义的模型类
import matplotlib.pyplot as plt

# --------- 参数配置 ---------
MODEL_PATH = 'model.pth'
DATA_PATH = 'product_data.csv'  # CSV 文件应包含商品 ID 和其时间序列特征
TOP_K = 10
INPUT_SIZE = 72  # 每个商品的时间序列长度
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------- 1. 加载数据 ---------
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


feature_data = df.iloc[:, 1:].values
# product_name_list 是df的列名
product_name_list = df.columns[1:].tolist()


# --------- 2. 数据标准化 ---------
scaler = StandardScaler()
feature_data_scaled = scaler.fit_transform(feature_data)

# 转为张量
input_tensor = torch.tensor(feature_data_scaled, dtype=torch.float32).to(DEVICE)

# --------- 3. 加载模型 ---------
model = DualHeadBiLSTM(input_size=INPUT_SIZE, output_size=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------- 4. 预测 ---------
with torch.no_grad():
    _, cls_out = model(input_tensor)

cls_probs = cls_out.squeeze().cpu().numpy()  # shape: (num_products,)

# --------- 5. 获取 Top 10 涨价概率商品 ---------
top_indices = np.argsort(cls_probs)[-TOP_K:][::-1]

print("Top 10 products most likely to increase in price:\n")
top_items = []
for idx in top_indices:
    print(f"{product_name_list[idx]}: {cls_probs[idx]:.4f}")
    top_items.append((product_name_list[idx], cls_probs[idx]))

# --------- 6. 保存结果 ---------
df_top10 = pd.DataFrame(top_items, columns=["ProductID", "Increase_Probability"])
df_top10.to_csv("top10_price_increase_predictions.csv", index=False)

# --------- 7. 可视化 ---------
plt.figure(figsize=(10, 5))
plt.barh([x[0] for x in reversed(top_items)], [x[1] for x in reversed(top_items)], color='orange')
plt.xlabel("Probability of Price Increase")
plt.title("Top 10 Products Most Likely to Increase in Price")
plt.tight_layout()
plt.savefig("top10_prediction_plot.png")
plt.show()
