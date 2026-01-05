import os
import requests
import pandas as pd
import time
import schedule
import oss2
from datetime import datetime

# ================= 配置区域 =================
# 从环境变量获取 AK/SK
OSS_AK = os.getenv('demos_oss_ak')
OSS_SK = os.getenv('demos_oss_sk')

OSS_BUCKET_NAME = 'milky-way-idle-oss'  # 你的 Bucket
OSS_ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'  # 你的 Endpoint

MARKET_API_URL = "https://www.milkywayidle.com/game_data/marketplace.json"

# 【新配置】本地数据存放文件夹名称
LOCAL_DATA_DIR = 'market_data'


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


def fetch_and_store_data():
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')

    # 1. 确保目录存在
    if not os.path.exists(LOCAL_DATA_DIR):
        os.makedirs(LOCAL_DATA_DIR)

    # 2. 构造文件名和完整路径
    # 文件名: market_2026-01-05.csv.gz
    filename = f"market_{date_str}.csv.gz"
    # 路径: market_data/market_2026-01-05.csv.gz
    local_full_path = os.path.join(LOCAL_DATA_DIR, filename)

    print(f"[{now.strftime('%H:%M:%S')}] Fetching...", end=" ")

    try:
        response = requests.get(MARKET_API_URL, timeout=10)
        data = response.json()
        market_data = data.get("marketData", {})
        timestamp = data.get("timestamp")

        if not market_data:
            print("Empty data.")
            return

        records = []
        for item_key, levels in market_data.items():
            clean_name = item_key.replace("/items/", "")
            for level, prices in levels.items():
                records.append({
                    "t": timestamp,
                    "i": clean_name,
                    "l": int(level),
                    "a": prices.get('a', -1),
                    "b": prices.get('b', -1)
                })

        if not records: return

        new_df = pd.DataFrame(records)

        # 检查文件是否存在（决定是否写表头）
        file_exists = os.path.isfile(local_full_path)

        # 3. 写入到指定目录下的文件中
        new_df.to_csv(
            local_full_path,
            mode='a',
            header=not file_exists,
            index=False,
            compression='gzip'
        )

        print(f"Saved {len(records)} rows to {local_full_path}.")

        # 4. 上传到 OSS
        # 注意：local_path 传完整路径，oss_path 依然保持原来的结构
        oss_path = f"milkyway/{now.year}/{now.month:02d}/{filename}"
        upload_to_oss(local_full_path, oss_path)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Milky Way Idle Crawler (GZIP + Folder) ===")
    print(f"Data will be saved to: ./{LOCAL_DATA_DIR}/")
    print("Waiting for next minute start (:00)...")

    # 严格在每分钟的第 00 秒执行
    schedule.every().minute.at(":00").do(fetch_and_store_data)

    while True:
        schedule.run_pending()
        time.sleep(0.1)
