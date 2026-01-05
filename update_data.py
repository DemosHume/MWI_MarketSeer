import os
import requests
import pandas as pd
import time
import schedule
import oss2
from datetime import datetime

# ================= 配置区域 =================

# 1. 本地文件名
LOCAL_FILE = 'market_history.csv'

# 2. 阿里云 OSS 配置
# 从环境变量获取 AK/SK
OSS_AK = os.getenv('demos_oss_ak')
OSS_SK = os.getenv('demos_oss_sk')

# 【请修改这里】你的 Bucket 名字
OSS_BUCKET_NAME = 'milky-way-idle-oss'

# 【请修改这里】你的 Endpoint (例如杭州是 oss-cn-hangzhou.aliyuncs.com)
OSS_ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'

# OSS 上保存的文件路径 (Key)
OSS_OBJECT_KEY = 'milkyway/market_history.csv'

# 3. 游戏 API 地址
MARKET_API_URL = "https://www.milkywayidle.com/game_data/marketplace.json"


# ===========================================

def upload_to_oss(local_path):
    """
    将文件上传到阿里云 OSS
    """
    if not OSS_AK or not OSS_SK:
        print(">>> [OSS Error] 环境变量 demos_oss_ak 或 demos_oss_sk 未设置，跳过上传。")
        return

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Uploading to OSS: {OSS_OBJECT_KEY} ...")

        # 1. 认证
        auth = oss2.Auth(OSS_AK, OSS_SK)

        # 2. 初始化 Bucket
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

        # 3. 上传文件 (put_object_from_file 适合中小文件，如果文件超大可用 multipart)
        bucket.put_object_from_file(OSS_OBJECT_KEY, local_path)

        print(f">>> [OSS Success] Upload completed.")

    except oss2.exceptions.OssError as e:
        print(f">>> [OSS Error] {e.message}")
    except Exception as e:
        print(f">>> [OSS Error] {e}")


def fetch_and_store_data():
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now_str}] Fetching market data...")

    try:
        # 1. 请求接口
        response = requests.get(MARKET_API_URL, timeout=30)
        data = response.json()

        market_data = data.get("marketData", {})
        timestamp = data.get("timestamp")

        if not market_data or not timestamp:
            print("No data received.")
            return

        dt_record = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        # 2. 解析数据 (转为长表格式)
        records = []
        for item_key, levels in market_data.items():
            clean_name = item_key.replace("/items/", "")
            for level, prices in levels.items():
                # 同时获取 a (ask) 和 b (bid)
                # 如果没有挂单，API可能不返回 key，这里默认给 -1
                ask = prices.get('a', -1)
                bid = prices.get('b', -1)

                records.append({
                    "timestamp": timestamp,
                    "datetime": dt_record,
                    "item": clean_name,
                    "level": int(level),
                    "ask": ask,  # 卖一价
                    "bid": bid  # 买一价
                })

        if not records:
            return

        new_df = pd.DataFrame(records)

        # 3. 追加写入 CSV (高效写入)
        # 如果文件不存在，写入表头；如果存在，不写入表头直接追加
        file_exists = os.path.isfile(LOCAL_FILE)

        new_df.to_csv(
            LOCAL_FILE,
            mode='a',
            header=not file_exists,
            index=False
        )

        print(f"[{now_str}] Saved {len(records)} rows locally.")

        # 4. 触发 OSS 上传
        # 注意：这里是同步上传，如果网络慢会阻塞几秒。
        # 考虑到是 5分钟一次，通常没问题。
        upload_to_oss(LOCAL_FILE)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    print("=== Milky Way Idle Market Watcher Started ===")
    print(f"Local File: {LOCAL_FILE}")
    print(f"Target OSS: {OSS_BUCKET_NAME} -> {OSS_OBJECT_KEY}")

    # 启动时先跑一次
    fetch_and_store_data()

    # 设定定时任务 (每5分钟)
    schedule.every(5).minutes.do(fetch_and_store_data)

    while True:
        schedule.run_pending()
        time.sleep(1)
