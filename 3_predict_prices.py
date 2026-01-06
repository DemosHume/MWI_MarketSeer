import pandas as pd
import joblib
import os
import numpy as np
from colorama import init, Fore
from tqdm import tqdm

init(autoreset=True)

from utils.utils import load_clean_data, extract_features, prepare_item_data, prepare_predict_data

# === 配置 ===
DATA_PATH = 'data/market_history.csv'
MODEL_DIR = 'models'

def predict_all():
    # 1. 加载最新数据并提取特征
    print(Fore.CYAN + "正在加载数据...")
    pivot_df = load_clean_data(DATA_PATH)
    if pivot_df is None or len(pivot_df) < 5:
        print(Fore.RED + "数据不足，无法预测")
        return

    returns_df, market_features = extract_features(pivot_df, n_components=5)
    
    # 2. 扫描所有模型
    if not os.path.exists(MODEL_DIR):
        print(Fore.RED + "模型目录不存在，请先运行 2_train_model.py")
        return
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    if not model_files:
        print(Fore.RED + "没有找到任何训练好的模型")
        return

    print(f"找到 {len(model_files)} 个模型，正在进行批量预测...")
    
    # 获取最新的 bid 价格用于差价过滤
    raw_df = pd.read_csv(DATA_PATH)
    raw_df['id'] = raw_df['item'] + "_lv" + raw_df['level'].astype(str)
    latest_ts = pivot_df.index[-1]
    latest_bids = raw_df[raw_df['timestamp'] == latest_ts].set_index('id')['bid'].to_dict()

    results = []
    filtered_by_spread = 0
    
    # 获取最后一个时间点的特征
    pbar = tqdm(model_files, desc="批量预测", unit="model")
    for f in pbar:
        item_id = f.replace('.pkl', '')
        model_path = os.path.join(MODEL_DIR, f)
        
        try:
            model = joblib.load(model_path)
            
            # 1. 准备验证数据 (含有已知 target 的最后一行)
            data = prepare_item_data(item_id, returns_df, market_features)
            if data is None or len(data) == 0:
                continue
            
            # 验证预测 (对已知的最后一次变动进行预测)
            val_X = data.drop(columns=['target']).iloc[[-1]]
            val_actual = data['target'].iloc[-1]
            val_pred = model.predict(val_X)[0]
            val_error = val_pred - val_actual

            # 2. 准备未来预测特征 (最新的特征，尚无 target)
            latest_X_future = prepare_predict_data(item_id, returns_df, market_features)
            if latest_X_future is None:
                continue
                
            pred_future_return = model.predict(latest_X_future)[0]
            
            # 获取当前价格 (Ask)
            current_price = pivot_df[item_id].iloc[-1]
            bid_price = latest_bids.get(item_id, -1)
            
            # 3. 差价过滤: 买卖价差异超过 20% 的不显示 (Ask > Bid * 1.2)
            # 如果没有买单 (bid <= 0) 或差价太大，则跳过
            if bid_price <= 0:
                filtered_by_spread += 1
                continue
            
            spread_ratio = (current_price - bid_price) / bid_price
            if spread_ratio > 0.2:
                filtered_by_spread += 1
                continue
            
            results.append({
                'item_id': item_id,
                'current_price': current_price,
                'val_actual_pct': val_actual * 100,
                'val_pred_pct': val_pred * 100,
                'val_err_pct': abs(val_error) * 100,
                'pred_future_pct': pred_future_return * 100
            })
        except Exception as e:
            # print(f"预测 {item_id} 出错: {e}")
            continue

    # 3. 排序并展示结果
    if filtered_by_spread > 0:
        print(Fore.YELLOW + f"提示: 已从排行榜中剔除 {filtered_by_spread} 个买卖价差过大(>20%)的商品")
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print(Fore.YELLOW + "没有预测结果")
        return
        
    # 按预测的未来涨幅排序
    res_df = res_df.sort_values('pred_future_pct', ascending=False)

    print("\n" + Fore.GREEN + "=== 市场预测与验证榜 (Top 10 潜力股) ===")
    # 增加一列显示验证误差，让用户知道模型在最新数据上的表现
    header = f"{'物品ID':<25} | {'当前价':>8} | {'预测涨幅':>8} | {'上次误差':>8}"
    print(header)
    print("-" * len(header))
    
    for _, row in res_df.head(10).iterrows():
        color = Fore.GREEN if row['pred_future_pct'] > 0 else Fore.WHITE
        print(color + f"{row['item_id']:<25} | {row['current_price']:>8.1f} | {row['pred_future_pct']:>7.2f}% | {row['val_err_pct']:>7.2f}%")

    print("\n" + Fore.RED + "=== 市场预测跌幅榜 (Top 10 避坑) ===")
    for _, row in res_df.tail(10).iterrows():
        color = Fore.RED if row['pred_future_pct'] < 0 else Fore.WHITE
        print(color + f"{row['item_id']:<25} | {row['current_price']:>8.1f} | {row['pred_future_pct']:>7.2f}% | {row['val_err_pct']:>7.2f}%")

    # 打印总体验证表现
    avg_val_err = res_df['val_err_pct'].mean()
    print(f"\n全市场模型平均验证误差 (最近1期): {avg_val_err:.2f}%")

if __name__ == "__main__":
    predict_all()
