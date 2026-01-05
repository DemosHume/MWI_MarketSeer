import pandas as pd
import numpy as np
import joblib
import os
from colorama import init, Fore

init(autoreset=True)

# === 配置 ===
DATA_PATH = 'data/market_history.csv'
MODEL_PATH = 'models/abyssal_essence_lv0.pkl'  # 对应刚才训练的模型
TARGET_ITEM = 'abyssal_essence'
TARGET_LEVEL = 0


def predict_future():
    # 1. 加载模型
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + "模型文件不存在，请先运行 2_train_model.py")
        return

    model = joblib.load(MODEL_PATH)

    # 2. 加载最新数据
    df = pd.read_csv(DATA_PATH)

    # 3. 提取特定物品的最后一段数据来构造特征
    df_item = df[(df['item'] == TARGET_ITEM) & (df['level'] == TARGET_LEVEL)].copy()

    # 数据清洗
    df_item['ask'] = df_item['ask'].replace(-1, np.nan).ffill().fillna(0)

    # 我们需要最后一行数据来进行预测
    # 但是为了计算 MA12，我们需要至少最后12行数据
    if len(df_item) < 15:
        print(Fore.RED + "数据量太少，无法计算指标")
        return

    # 重新计算特征 (逻辑必须与训练时完全一致!)
    last_rows = df_item.tail(20).copy()  # 取最后20行计算指标

    last_rows['lag_1'] = last_rows['ask'].shift(1)
    last_rows['lag_2'] = last_rows['ask'].shift(2)
    last_rows['lag_3'] = last_rows['ask'].shift(3)
    last_rows['MA5'] = last_rows['ask'].rolling(window=5).mean()
    last_rows['MA12'] = last_rows['ask'].rolling(window=12).mean()
    last_rows['std_5'] = last_rows['ask'].rolling(window=5).std()

    # 获取最后一行 (包含最新时刻的所有特征)
    latest_features = last_rows.iloc[[-1]][['ask', 'lag_1', 'lag_2', 'lag_3', 'MA5', 'MA12', 'std_5']]

    current_price = latest_features.iloc[0]['ask']
    current_time = last_rows.iloc[-1]['datetime']

    # 4. 预测
    predicted_price = model.predict(latest_features)[0]

    # 5. 输出决策建议
    change_percent = ((predicted_price - current_price) / current_price) * 100

    print(Fore.CYAN + "=" * 30)
    print(f"物品: {TARGET_ITEM} (Lv{TARGET_LEVEL})")
    print(f"数据时间: {current_time}")
    print(f"当前卖价: {current_price:,.0f}")
    print(Fore.YELLOW + f"预测下个周期价格: {predicted_price:,.0f}")

    print("-" * 20)
    if change_percent > 1.0:
        print(Fore.GREEN + f"🚀 趋势: 强力看涨 (+{change_percent:.2f}%)")
        print("建议: 考虑买入")
    elif change_percent < -1.0:
        print(Fore.RED + f"📉 趋势: 强力看跌 ({change_percent:.2f}%)")
        print("建议: 观望或抛售")
    else:
        print(Fore.WHITE + f"➡️ 趋势: 震荡 (波动 {change_percent:.2f}%)")
        print("建议: 保持持有")
    print(Fore.CYAN + "=" * 30)


if __name__ == "__main__":
    predict_future()
