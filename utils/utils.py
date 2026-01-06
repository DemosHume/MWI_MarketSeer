import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA


def load_clean_data(data_path):
    """加载并清洗数据，返回宽表形式"""
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        return None

    # 读取数据
    df = pd.read_csv(data_path)

    # 1. 处理时间格式
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 2. 处理无效价格 -1
    # 将 -1 替换为 NaN，方便后续识别
    if 'ask' in df.columns:
        df['ask'] = df['ask'].replace(-1, np.nan)

    # 【策略 1】价差过滤 (Spread Filter)
    # 过滤掉“毫无诚意”的报价（如 100万的苹果 或 5块钱的手机）
    # 在正常市场中，Ask 应该略高于 Bid。如果两者比例失调，说明数据异常。
    # 用户指出：Ask 不可能比 Bid 高出很多倍（会自动成交），Bid 超过 Ask 太多也是异常。
    if 'ask' in df.columns and 'bid' in df.columns:
        max_ratio = 1.5  # 最大允许的价差比例
        # 只要 Ask/Bid 或 Bid/Ask 超过阈值，就视为异常
        bad_spread_mask = (df['bid'] > 0) & (df['ask'] > 0) & \
                         ((df['ask'] > df['bid'] * max_ratio) | (df['bid'] > df['ask'] * max_ratio))
        df.loc[bad_spread_mask, 'ask'] = np.nan

    # 3. 去重
    if 'timestamp' in df.columns and 'item' in df.columns:
        df = df.sort_values(['timestamp', 'item', 'level'])
        df = df.drop_duplicates(subset=['timestamp', 'item', 'level'], keep='last')

    # 4. 转换为宽表
    df['id'] = df['item'] + "_lv" + df['level'].astype(str)

    # index是时间，columns是物品ID，values是ask价格
    pivot_df = df.pivot(index='timestamp', columns='id', values='ask')

    # 【策略 2】统计偏离过滤 (Rolling Median)
    # 如果当前价格比过去一段时间的中位数高出/低于太多，认为是异常值（插针）
    if not pivot_df.empty:
        # 使用滚动中位数，窗口大小 5，居中，最少 1 个样本即可计算
        rolling_median = pivot_df.rolling(window=5, center=True, min_periods=1).median()
        # 偏离倍数阈值 (3倍)
        max_dev_ratio = 3
        deviation_mask = (pivot_df > rolling_median * max_dev_ratio) | \
                         (pivot_df < rolling_median / max_dev_ratio)
        pivot_df[deviation_mask] = np.nan

    # ==========================================
    # 【新增逻辑】严格过滤：剔除包含无效记录 (NaN/插针) 的物品
    # ==========================================
    original_count = pivot_df.shape[1]

    # dropna(axis=1) 表示：如果这一列(axis=1)里有任何一个NaN，就删掉整列
    pivot_df = pivot_df.dropna(axis=1, how='any')

    filtered_count = pivot_df.shape[1]
    print(f"数据清洗: 已剔除包含缺货记录的物品 {original_count - filtered_count} 个，剩余 {filtered_count} 个稳定物品。")

    # 5. 检查数据长度
    # 如果行数太少，特征工程计算完就没有数据了
    min_required_rows = 15  # 至少需要15条记录才能跑起来(Lag + Horizon + Buffer)
    if len(pivot_df) < min_required_rows:
        print(f"⚠️ 警告: 当前数据只有 {len(pivot_df)} 行。")
        print(f"   模型训练至少需要 {min_required_rows} 行数据（约运行15分钟）。")
        print("   >>> 请继续运行爬虫积累数据，稍后再试。 <<<")
        # 这里不返回 None，而是返回空表或者少量的表，让后续流程自己处理，但通常后面会训练出0个模型

    return pivot_df


def extract_features(pivot_df, target_id=None, n_components=5):
    """
    特征工程：
    1. 计算收益率 (Percentage Change)
    2. 计算全市场平均涨跌幅 (Market Return)
    3. PCA 降维捕获物品间相关性
    """
    # === 关键修复开始 ===
    # 计算收益率
    returns_df = pivot_df.pct_change()

    # 1. 将无穷大 (inf/-inf) 替换为 NaN
    # (当价格从0变到有价格，或者有价格变到0时，会出现无限大)
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)

    # 2. 将 NaN 填充为 0
    returns_df = returns_df.fillna(0)
    # === 关键修复结束 ===

    # 全市场平均收益率 (代表大盘走势)
    market_return = returns_df.mean(axis=1)

    # PCA 降维：捕获物品之间的潜在相关性 (如原材料和成品的共变)
    # 只有当样本量(行数)和特征量(列数)都大于 n_components 时才能做 PCA
    pca_df = None

    # 动态调整组件数量，防止数据太少时报错
    valid_components = min(n_components, returns_df.shape[0], returns_df.shape[1])

    if valid_components > 0:
        try:
            pca = PCA(n_components=valid_components)
            pca_features = pca.fit_transform(returns_df)

            # 生成列名
            pca_cols = [f'pca_{i}' for i in range(valid_components)]
            pca_df = pd.DataFrame(pca_features, index=returns_df.index, columns=pca_cols)

            # 如果实际组件少于请求的组件，补齐列（用0填充）以保持格式一致
            if valid_components < n_components:
                for i in range(valid_components, n_components):
                    pca_df[f'pca_{i}'] = 0
        except Exception as e:
            print(f"PCA warning: {e}")
            # 如果PCA失败，生成全0矩阵
            pca_df = pd.DataFrame(0, index=returns_df.index, columns=[f'pca_{i}' for i in range(n_components)])
    else:
        # 数据不足，生成全0矩阵
        pca_df = pd.DataFrame(0, index=returns_df.index, columns=[f'pca_{i}' for i in range(n_components)])

    # 构造基础特征池
    base_features = pd.concat([market_return.rename('market_ret'), pca_df], axis=1)

    return returns_df, base_features


def prepare_item_data(target_id, returns_df, base_features, lookback=3, horizon=1):
    """为特定物品准备训练数据 (带调试打印版)"""
    if target_id not in returns_df.columns:
        return None

    # 1. 目标变量
    y = returns_df[target_id].shift(-horizon)

    # 2. 滞后特征
    item_ret = returns_df[target_id]
    lags_list = [item_ret.shift(i).rename(f'lag_{i}') for i in range(lookback)]
    item_lags = pd.concat(lags_list, axis=1)

    # 3. 合并所有特征
    # 注意：这里我们先不 dropna，而是先打印看看
    X = pd.concat([item_lags, base_features], axis=1)
    raw_data = pd.concat([X, y.rename('target')], axis=1)

    # === 🕵️‍♂️ 侦探代码 开始 ===
    # 只针对第一个物品打印调试信息
    if target_id == returns_df.columns[0]:
        print(f"\n======== DEBUG: {target_id} 数据诊断 ========")
        print(f"原始数据行数: {len(raw_data)}")

        # 检查各部分是否有全空的情况
        print(f"Lag特征空值数: {item_lags.isna().sum().sum()} (正常应该是 {lookback} * 列数)")
        print(f"Base特征空值数: {base_features.isna().sum().sum()} (这里应该接近 0)")
        print(f"Target空值数: {y.isna().sum()}")

        # 检查合并后的空值分布
        null_rows = raw_data.isna().any(axis=1).sum()
        print(f"包含空值的行数: {null_rows}")
        print(f"完全清洗后剩余行数: {len(raw_data) - null_rows}")

        # 打印前10行看看长什么样
        print("\n--- 数据预览 (前5行) ---")
        print(raw_data.head(5))
        print("==========================================\n")
    # === 🕵️‍♂️ 侦探代码 结束 ===

    # 4. 正式清洗
    data = raw_data.dropna()

    if len(data) == 0:
        return None

    return data
