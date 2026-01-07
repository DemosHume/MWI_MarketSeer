import joblib
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from colorama import init, Fore
from tqdm import tqdm
from utils.utils import load_clean_data, extract_features, prepare_item_data

init(autoreset=True)

# === 配置 ===
DATA_PATH = 'data/market_history.csv'
MODEL_DIR = 'models'
MIN_SAMPLES = 10  # 恢复到较小值以便在初期数据上也能训练
VAL_ROWS = 60      # 使用最后 1 小时的数据作为验证集 (假设 1min/row)
MAX_ITEMS = None  # 设置为 None 则训练所有物品

def train_all_models():
    print(Fore.CYAN + "正在加载并预处理市场全量数据...")
    pivot_df = load_clean_data(DATA_PATH)
    
    if pivot_df is None or len(pivot_df) < 5:
        print(Fore.RED + "数据不足，无法提取特征。请确保 market_history.csv 有足够的时间点。")
        return

    # 提取全局特征 (PCA 相关性 + 市场大盘)
    returns_df, market_features = extract_features(pivot_df, n_components=5)
    
    # 获取所有物品ID
    all_item_ids = pivot_df.columns.tolist()
    if MAX_ITEMS:
        all_item_ids = all_item_ids[:MAX_ITEMS]

    print(f"准备训练 {len(all_item_ids)} 个物品的模型...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"DEBUG: 时间序列总行数: {len(pivot_df)}")
    if len(pivot_df) < 15:
        print(f"❌ 严重警告: 数据行数只有 {len(pivot_df)} 行！")
        print(f"👉 建议: 请让爬虫继续运行一段时间，建议积累 30 行以上数据后再训练。")

    success_count = 0
    skip_insufficient_samples = 0
    skip_no_volatility = 0
    skip_error = 0
    total_mae = 0
    
    pbar = tqdm(all_item_ids, desc="训练模型", unit="item")
    for item_id in pbar:
        try:
            # 准备该物品的数据
            data = prepare_item_data(item_id, returns_df, market_features)
            
            if data is None or len(data) < MIN_SAMPLES:
                skip_insufficient_samples += 1
                continue
            
            # 检查是否有波动
            if data['target'].nunique() <= 1:
                skip_no_volatility += 1
                continue

            X = data.drop(columns=['target'])
            y = data['target']

            # 训练模型 (保留最后 VAL_ROWS 行用于验证)
            # 如果总样本数太少，自动调整验证集大小
            current_val_size = min(VAL_ROWS, len(data) // 4)
            if current_val_size < 1: current_val_size = 1
            
            X_train = X.iloc[:-current_val_size]
            y_train = y.iloc[:-current_val_size]
            X_test = X.iloc[-current_val_size:]
            y_test = y.iloc[-current_val_size:]
            
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='mae',
                early_stopping_rounds=10
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # 验证性能
            preds = model.predict(X_test)
            mae = np.mean(np.abs(preds - y_test))
            total_mae += mae
            success_count += 1

            # 保存
            model_path = os.path.join(MODEL_DIR, f'{item_id}.pkl')
            joblib.dump(model, model_path)
            
            # 更新进度条描述
            if success_count % 10 == 0:
                pbar.set_postfix({"成功": success_count, "Avg MAE": f"{total_mae/success_count:.4f}"})

        except Exception as e:
            skip_error += 1
            continue

    print(Fore.GREEN + f"\n=== 训练完成 ===")
    print(f"成功训练物品数: {success_count}")
    if success_count > 0:
        print(f"平均验证误差 (MAE): {total_mae / success_count:.6f}")
    
    print(f"跳过详情:")
    print(f"  - 样本不足 (<{MIN_SAMPLES}行): {skip_insufficient_samples}")
    print(f"  - 价格无波动: {skip_no_volatility}")
    print(f"  - 训练报错: {skip_error}")
    print(f"模型保存在: {MODEL_DIR}")

if __name__ == "__main__":
    train_all_models()
