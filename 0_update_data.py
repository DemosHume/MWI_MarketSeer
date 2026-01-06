import os
import requests
import pandas as pd
import time
import schedule
import oss2
from datetime import datetime

# ================= 配置区域 =================
OSS_AK = os.getenv('mwi_oss_ak')
OSS_SK = os.getenv('mwi_oss_sk')

OSS_BUCKET_NAME = 'milky-way-idle-oss'
OSS_ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'

MARKET_API_URL = "https://www.milkywayidle.com/game_data/marketplace.json"
LOCAL_DATA_DIR = 'market_data'

# ================= 全局变量 =================
# 记录上一次处理的时间戳
last_processed_timestamp = None


# ===========================================

def upload_to_oss(local_path, oss_path):
    """上传文件到 OSS"""
    if not OSS_AK or not OSS_SK:
        return
    try:
        auth = oss2.Auth(OSS_AK, OSS_SK)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
        bucket.put_object_from_file(oss_path, local_path)
    except Exception as e:
        print(f">>> [OSS Error] {e}")


def get_latest_timestamp_from_file(filepath):
    """
    高效读取 GZIP CSV 文件的最后一行，获取时间戳
    """
    if not os.path.exists(filepath):
        return None

    try:
        # 使用 chunksize 避免一次性加载整个文件到内存
        # 我们只关心 't' (timestamp) 这一列
        chunk_iter = pd.read_csv(
            filepath,
            compression='gzip',
            usecols=['t'],
            chunksize=1000
        )

        last_chunk = None
        for chunk in chunk_iter:
            last_chunk = chunk

        if last_chunk is not None and not last_chunk.empty:
            # 返回最后一行的 t 值
            return last_chunk['t'].iloc[-1]

    except Exception as e:
        print(f"[Warning] Failed to read local file timestamp: {e}")

    return None


def fetch_and_store_data():
    global last_processed_timestamp

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')

    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)

    filename = f"market_{date_str}.csv.gz"
    local_full_path = os.path.join(LOCAL_DATA_DIR, filename)

    print(f"[{now.strftime('%H:%M:%S')}] Checking...", end=" ")

    # ========================================================
    # 核心逻辑：如果是脚本刚启动（None），先尝试从本地文件恢复状态
    # ========================================================
    if last_processed_timestamp is None:
        last_processed_timestamp = get_latest_timestamp_from_file(local_full_path)
        if last_processed_timestamp:
            ts_str = datetime.fromtimestamp(int(last_processed_timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[Init] Loaded last timestamp from file: {last_processed_timestamp} ({ts_str})")
        else:
            print(f"[Init] No local data found, starting fresh.")

    try:
        response = requests.get(MARKET_API_URL, timeout=10)
        data = response.json()

        current_data_timestamp = data.get("timestamp")

        # 对比 API 时间戳 和 本地/内存记录的时间戳
        if last_processed_timestamp is not None and current_data_timestamp == last_processed_timestamp:
            ts_str = datetime.fromtimestamp(int(current_data_timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Skipped. (Timestamp {current_data_timestamp} ({ts_str}) already exists)")
            return

        market_data = data.get("marketData", {})
        if not market_data:
            print("Empty data.")
            return

        records = []
        for item_key, levels in market_data.items():
            clean_name = item_key.replace("/items/", "")
            for level, prices in levels.items():
                records.append({
                    "t": current_data_timestamp,
                    "i": clean_name,
                    "l": int(level),
                    "a": prices.get('a', -1),
                    "b": prices.get('b', -1)
                })

        if not records: return

        new_df = pd.DataFrame(records)

        file_exists = os.path.isfile(local_full_path)

        new_df.to_csv(
            local_full_path,
            mode='a',
            header=not file_exists,
            index=False,
            compression='gzip'
        )

        ts_str = datetime.fromtimestamp(int(current_data_timestamp)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Saved {len(records)} rows. (TS: {current_data_timestamp} ({ts_str}))")

        # 更新内存记录
        last_processed_timestamp = current_data_timestamp

        # 上传到 OSS
        oss_path = f"milkyway/{now.year}/{now.month:02d}/{filename}"
        upload_to_oss(local_full_path, oss_path)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Milky Way Idle Crawler (Local State Check) ===")
    print(f"Data directory: ./{LOCAL_DATA_DIR}/")

    # 立即运行一次，以便初始化 last_processed_timestamp
    # 这样启动时就能立刻知道当前本地文件的状态，不用等下一分钟
    fetch_and_store_data()

    print("Waiting for next minute start (:00)...")

    schedule.every().minute.at(":00").do(fetch_and_store_data)

    while True:
        schedule.run_pending()
        time.sleep(0.1)
