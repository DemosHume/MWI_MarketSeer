import pandas as pd
import joblib
import os
from colorama import init, Fore

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
    for f in model_files:
        item_id = f.replace('.pkl', '')
        model_path = os.path.join(MODEL_DIR, f)
        
        try:
            model = joblib.load(model_path)
            
            # 准备该物品的最新特征行
            data = prepare_item_data(item_id, returns_df, market_features)
            if data is None or len(data) == 0:
                continue
                
            # 取最后一行特征进行预测
            latest_X = data.drop(columns=['target']).iloc[[-1]]
            pred_return = model.predict(latest_X)[0]
            
            # 获取当前价格
            current_price = pivot_df[item_id].iloc[-1]
            
            results.append({
                'item_id': item_id,
                'current_price': current_price,
                'pred_return_pct': pred_return * 100
            })
        except Exception as e:
            continue

    # 3. 排序并展示结果
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print(Fore.YELLOW + "没有预测结果")
        return
        
    res_df = res_df.sort_values('pred_return_pct', ascending=False)

    print("\n" + Fore.GREEN + "=== 市场预测涨跌榜 (Top 10 看涨) ===")
    print(f"{'物品ID':<30} | {'当前价':<10} | {'预测涨幅':<10}")
    print("-" * 55)
    
    for _, row in res_df.head(10).iterrows():
        color = Fore.GREEN if row['pred_return_pct'] > 0 else Fore.WHITE
        print(color + f"{row['item_id']:<30} | {row['current_price']:>10.1f} | {row['pred_return_pct']:>9.2f}%")

    print("\n" + Fore.RED + "=== 市场预测跌幅榜 (Top 10 看跌) ===")
    for _, row in res_df.tail(10).iterrows():
        color = Fore.RED if row['pred_return_pct'] < 0 else Fore.WHITE
        print(color + f"{row['item_id']:<30} | {row['current_price']:>10.1f} | {row['pred_return_pct']:>9.2f}%")

if __name__ == "__main__":
    predict_all()
