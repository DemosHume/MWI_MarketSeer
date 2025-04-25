import os
import sqlite3
import pandas as pd
from datetime import datetime
import pytz
import requests
import json

from tqdm import tqdm


def query(q_string, fname='market.db'):
    conn = sqlite3.connect(fname)
    data = pd.read_sql_query(q_string, conn)
    conn.close()
    return data


def timestamp_to_beijing_time(ts):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(beijing_tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def get_github_latest_commit_hash(owner, repo, branch="main"):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data['sha']
    else:
        raise Exception(f"GitHub API error: {response.status_code} - {response.text}")


def read_stored_hash(file_path):
    """从 JSON 文件中读取存储的上次更新的仓库 commit hash"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('commit_hash', None)
    return None


def store_hash(file_path, hash_value):
    """将当前仓库的 commit hash 存储到 JSON 文件"""
    data = {'commit_hash': hash_value}
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def refresh_market_data():
    url = "https://github.com/holychikenz/MWIApi/raw/main/market.db"
    proxy = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    }
    output_file = "market.db"
    hash_file = "market_data_info.json"

    # 获取最新的 commit hash
    current_hash = get_github_latest_commit_hash("holychikenz", "MWIApi")
    stored_hash = read_stored_hash(hash_file)

    # 如果 hash 一样，就跳过更新
    if current_hash == stored_hash:
        print("仓库没有变化，无需更新 ✅")
        latest_time = query('SELECT time FROM ask ORDER BY time DESC LIMIT 1')['time'].iloc[0]
        print(f"最新数据\t北京时间: {timestamp_to_beijing_time(latest_time)}")
        return

    # 如果 hash 不同，执行更新
    print("仓库有变化，开始更新数据...")

    # 发起请求
    with requests.get(url, proxies=proxy, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 8192

        # tqdm 进度条包装器
        with open(output_file, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=output_file
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # 存储当前的 hash 到 JSON 文件
    store_hash(hash_file, current_hash)
    print("更新完成 ✅")
    latest_time = query('SELECT time FROM ask ORDER BY time DESC LIMIT 1')['time'].iloc[0]
    print(f"最新数据\t北京时间: {timestamp_to_beijing_time(latest_time)}")

