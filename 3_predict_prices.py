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
    milk_results = []
    filtered_by_spread = 0
    filtered_by_noise = 0
    
    # 获取最后一个时间点的特征
    pbar = tqdm(model_files, desc="批量预测", unit="model")
    for f in pbar:
        item_id = f.replace('.pkl', '')
        # 只要包含 milk_lv 就算是 Milk 商品 (兼容 milk_lv0 和 azure_milk_lv0)
        is_milk = 'milk_lv' in item_id
        model_path = os.path.join(MODEL_DIR, f)
        
        try:
            model = joblib.load(model_path)
            
            # 1. 准备验证数据 (使用最近多期评估模型稳定性)
            data = prepare_item_data(item_id, returns_df, market_features)
            if data is None or len(data) == 0:
                continue
            
            # 计算最近 10 期的平均误差 (MAE)
            eval_n = min(10, len(data))
            val_X_recent = data.drop(columns=['target']).iloc[-eval_n:]
            val_actual_recent = data['target'].iloc[-eval_n:]
            val_preds_recent = model.predict(val_X_recent)
            recent_mae = np.mean(np.abs(val_preds_recent - val_actual_recent))
            last_error = abs(val_preds_recent[-1] - val_actual_recent.iloc[-1])

            # 2. 准备未来预测特征 (最新的特征，尚无 target)
            latest_X_future = prepare_predict_data(item_id, returns_df, market_features)
            if latest_X_future is None:
                continue
                
            pred_future_return = model.predict(latest_X_future)[0]
            
            # 获取当前价格 (Ask)
            current_price = pivot_df[item_id].iloc[-1]
            bid_price = latest_bids.get(item_id, -1)
            
            # 3. 结果记录
            res_entry = {
                'item_id': item_id,
                'current_price': current_price,
                'val_err_pct': last_error * 100,
                'recent_mae_pct': recent_mae * 100,
                'pred_future_pct': pred_future_return * 100,
                'bid_price': bid_price,
                'no_model': False
            }

            if is_milk:
                milk_results.append(res_entry)

            # 4. 差价过滤: 买卖价差异超过 20% 的不入排行榜 (Ask > Bid * 1.2)
            if bid_price <= 0:
                filtered_by_spread += 1
                continue
            
            spread_ratio = (current_price - bid_price) / bid_price
            if spread_ratio > 0.2:
                filtered_by_spread += 1
                continue
            
            # 5. 信号强度过滤: 预测涨幅必须大于近期平均误差，否则视为噪音 (信噪比 > 1)
            # 这样可以确保推荐的商品，其预测的“势头”超过了模型的平均“抖动”
            if abs(pred_future_return) > recent_mae:
                results.append(res_entry)
            else:
                filtered_by_noise += 1
        except Exception as e:
            # print(f"预测 {item_id} 出错: {e}")
            continue

    # 3. 补全缺失的 Milk 类商品 (即使没有模型也显示，用于反馈无波动商品)
    all_milk_ids = [col for col in pivot_df.columns if 'milk_lv' in col]
    processed_milk_ids = {r['item_id'] for r in milk_results}
    for milk_id in all_milk_ids:
        if milk_id not in processed_milk_ids:
            current_price = pivot_df[milk_id].iloc[-1]
            bid_price = latest_bids.get(milk_id, -1)
            milk_results.append({
                'item_id': milk_id,
                'current_price': current_price,
                'val_err_pct': 0.0,
                'recent_mae_pct': 0.0,
                'pred_future_pct': 0.0,
                'bid_price': bid_price,
                'no_model': True
            })

    # 4. 排序并展示结果
    if filtered_by_spread > 0 or filtered_by_noise > 0:
        msg = f"提示: 已剔除 {filtered_by_spread} 个大差价商品"
        if filtered_by_noise > 0:
            msg += f", {filtered_by_noise} 个低信噪比(预测幅度 < 误差)商品"
        print(Fore.YELLOW + msg)
        
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print(Fore.YELLOW + "没有可靠的预测结果 (当前所有商品预测幅度均小于模型误差)")
    else:
        # 按预测的未来涨幅排序
        res_df = res_df.sort_values('pred_future_pct', ascending=False)

        print("\n" + Fore.GREEN + "=== 市场预测与验证榜 (Top 潜力股) ===")
        # 增加一列显示验证误差，让用户知道模型在最新数据上的表现
        header = f"{'物品ID':<25} | {'当前价':>8} | {'预测涨幅':>8} | {'近期均误':>8}"
        print(header)
        print("-" * len(header))
        
        top_n = min(10, len(res_df))
        for _, row in res_df.head(top_n).iterrows():
            if row['pred_future_pct'] <= 0: continue # 只显示正增长的
            color = Fore.GREEN
            print(color + f"{row['item_id']:<25} | {row['current_price']:>8.1f} | {row['pred_future_pct']:>7.2f}% | {row['recent_mae_pct']:>7.2f}%")

        print("\n" + Fore.RED + "=== 市场预测跌幅榜 (Top 避坑) ===")
        for _, row in res_df.tail(top_n).iterrows():
            if row['pred_future_pct'] >= 0: continue # 只显示负增长的
            color = Fore.RED
            print(color + f"{row['item_id']:<25} | {row['current_price']:>8.1f} | {row['pred_future_pct']:>7.2f}% | {row['recent_mae_pct']:>7.2f}%")

    # 4. 展示 Milk 类商品专项预测
    if milk_results:
        milk_df = pd.DataFrame(milk_results).sort_values('pred_future_pct', ascending=False)
        # 确保 header 已经定义 (即使 res_df.empty 为真)
        header = f"{'物品ID':<25} | {'当前价':>8} | {'预测涨幅':>8} | {'近期均误':>8}"
        print("\n" + Fore.MAGENTA + "=== Milk 类商品专项预测 ===")
        print(header)
        print("-" * len(header))
        for _, row in milk_df.iterrows():
            # 检查是否因为差价大而被排行榜剔除
            is_filtered_spread = row['bid_price'] <= 0 or (row['current_price'] - row['bid_price']) / row['bid_price'] > 0.2
            is_noise = abs(row['pred_future_pct']) <= row['recent_mae_pct']
            no_model = row.get('no_model', False)
            
            prefix = "    "
            if is_filtered_spread: prefix = "[!] "
            elif is_noise and not no_model: prefix = "[~] "
            if no_model: prefix = "[?] "
            
            color = Fore.MAGENTA
            if not is_filtered_spread and not is_noise and not no_model:
                if row['pred_future_pct'] > 0: color = Fore.GREEN
                elif row['pred_future_pct'] < 0: color = Fore.RED
            
            val_err_str = f"{row['recent_mae_pct']:>7.2f}%" if not no_model else "  N/A   "
            print(color + f"{prefix}{row['item_id']:<21} | {row['current_price']:>8.1f} | {row['pred_future_pct']:>7.2f}% | {val_err_str}")
        
        has_filtered = any(row['bid_price'] <= 0 or (row['current_price'] - row['bid_price']) / row['bid_price'] > 0.2 for row in milk_results)
        has_noise = any(not row.get('no_model', False) and abs(row['pred_future_pct']) <= row['recent_mae_pct'] for row in milk_results)
        has_no_model = any(row.get('no_model', False) for row in milk_results)
        
        if has_filtered:
            print(Fore.YELLOW + "注: [!] 表示买卖价差过大(>20%)，预测仅供参考。")
        if has_noise:
            print(Fore.YELLOW + "注: [~] 表示信噪比低(预测幅度 < 误差)，预测不可靠。")
        if has_no_model:
            print(Fore.YELLOW + "注: [?] 表示该商品价格长期无波动，模型未训练。")

    # 打印总体验证表现
    if not res_df.empty:
        avg_val_err = res_df['recent_mae_pct'].mean()
        print(f"\n全市场模型平均验证误差 (最近10期): {avg_val_err:.2f}%")

if __name__ == "__main__":
    predict_all()
