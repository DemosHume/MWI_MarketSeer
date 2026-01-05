import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from colorama import init, Fore

init(autoreset=True)

# === 配置 ===
DATA_PATH = 'data/market_history.csv'
MODEL_DIR = 'models'
TARGET_ITEM = 'abyssal_essence'  # 你想预测的物品名称
TARGET_LEVEL = 0  # 物品等级
PREDICT_HORIZON = 1  # 预测未来第几个点（1代表预测5分钟后）


def prepare_features(df, item_name, level):
    """特征工程：将时间序列转换为监督学习数据"""
    # 1. 筛选特定物品
    df_item = df[(df['item'] == item_name) & (df['level'] == level)].copy()

    if df_item.empty:
        raise ValueError(f"没有找到 {item_name} Lv{level} 的数据")

    # 2. 数据清洗 (去除无效价格 -1, 替换为上一个有效值)
    df_item['ask'] = df_item['ask'].replace(-1, np.nan).ffill().fillna(0)

    # 3. 构造特征 (Feature Engineering)
    # 特征1: 滞后价格 (Lag features) - 过去3个点的价格
    for i in range(1, 4):
        df_item[f'lag_{i}'] = df_item['ask'].shift(i)

    # 特征2: 移动平均线 (Moving Averages)
    df_item['MA5'] = df_item['ask'].rolling(window=5).mean()
    df_item['MA12'] = df_item['ask'].rolling(window=12).mean()  # 1小时均线

    # 特征3: 波动率 (Standard Deviation)
    df_item['std_5'] = df_item['ask'].rolling(window=5).std()

    # 4. 构造目标变量 (Target) - 我们要预测未来的价格
    # shift(-1) 表示把下一行的价格拉到当前行作为 label
    df_item['target_price'] = df_item['ask'].shift(-PREDICT_HORIZON)

    # 5. 去除因为shift产生的空值行
    df_clean = df_item.dropna()

    return df_clean


def train_item_model():
    print(Fore.CYAN + "正在加载数据...")
    if not os.path.exists(DATA_PATH):
        print(Fore.RED + "数据文件不存在，请先运行 1_download_data.py")
        return

    df = pd.read_csv(DATA_PATH)

    try:
        # 准备数据
        data = prepare_features(df, TARGET_ITEM, TARGET_LEVEL)
        print(f"有效数据行数: {len(data)}")

        # 定义特征列和目标列
        feature_cols = ['ask', 'lag_1', 'lag_2', 'lag_3', 'MA5', 'MA12', 'std_5']
        X = data[feature_cols]
        y = data['target_price']

        # 划分训练集和测试集 (不打乱时间顺序)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 初始化模型 (XGBoost)
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

        print(Fore.YELLOW + f"开始训练 {TARGET_ITEM} 模型...")
        model.fit(X_train, y_train)

        # 评估
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        last_price = y_test.iloc[-1]

        print(Fore.GREEN + "=== 训练完成 ===")
        print(f"平均预测误差 (MAE): {mae:.2f}")
        print(f"误差百分比: {(mae / last_price) * 100:.2f}%")

        # 保存模型
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'{TARGET_ITEM}_lv{TARGET_LEVEL}.pkl')
        joblib.dump(model, model_path)
        print(Fore.GREEN + f"模型已保存至: {model_path}")

    except Exception as e:
        print(Fore.RED + f"训练出错: {e}")


if __name__ == "__main__":
    train_item_model()
