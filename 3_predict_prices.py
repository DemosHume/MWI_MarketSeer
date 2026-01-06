import pandas as pd
import joblib
import os
import numpy as np
from colorama import init, Fore
from tqdm import tqdm

init(autoreset=True)

from utils.utils import load_clean_data, extract_features, prepare_item_data

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
    
    results = []
    
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
            # 我们直接从 returns_df 和 market_features 拼接最新的一行
            # 这部分逻辑其实就是 prepare_item_data 内部 X 的逻辑，但取最后一行
            item_ret = returns_df[item_id]
            lookback = 3 # 保持与 prepare_item_data 一致
            
            # 手动构造最新一行的滞后特征
            latest_lags = [item_ret.iloc[-i-1] for i in range(lookback)]
            latest_market = market_features.iloc[-1].values
            
            # 拼接特征向量
            # 特征顺序: lag_0, lag_1, lag_2, market_ret, pca_0, pca_1, pca_2, pca_3, pca_4
            latest_feat_array = np.array(latest_lags + list(latest_market)).reshape(1, -1)
            
            # 将 numpy 数组转回 DataFrame 以保持特征名一致 (如果 XGBoost 训练时用了特征名)
            feature_names = [f'lag_{i}' for i in range(lookback)] + market_features.columns.tolist()
            latest_X_future = pd.DataFrame(latest_feat_array, columns=feature_names)
            
            pred_future_return = model.predict(latest_X_future)[0]
            
            # 获取当前价格
            current_price = pivot_df[item_id].iloc[-1]
            
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
