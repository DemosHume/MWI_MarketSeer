import os
import oss2
import pandas as pd
from glob import glob
from dotenv import load_dotenv
from colorama import init, Fore

init(autoreset=True)
load_dotenv()

# ================= 配置区域 =================
# OSS 配置
OSS_BUCKET_NAME = 'milky-way-idle-oss'
OSS_ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'

# 路径配置
LOCAL_TEMP_DIR = 'data/temp_daily'  # 临时存放下载的压缩包
FINAL_CSV_PATH = 'data/market_history.csv'  # 最终生成的单一整合大文件

# 数据保留配置
MAX_TIMESTAMPS = 500  # 至多保留最近多少个时间戳的数据，设置为 None 则保留所有


# ===========================================

def download_and_merge():
    ak = os.getenv('demos_oss_ak')
    sk = os.getenv('demos_oss_sk')

    if not ak or not sk:
        print(Fore.RED + "Error: 请在环境变量或 .env 设置 demos_oss_ak 和 demos_oss_sk")
        return

    # 1. 准备目录
    os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_CSV_PATH), exist_ok=True)

    print(Fore.CYAN + "=== 1. 开始从 OSS 下载增量数据 ===")
    try:
        auth = oss2.Auth(ak, sk)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

        # 遍历云端文件
        download_count = 0
        for obj in oss2.ObjectIterator(bucket, prefix='milkyway/'):
            if not obj.key.endswith('.csv.gz'):
                continue

            filename = os.path.basename(obj.key)
            local_path = os.path.join(LOCAL_TEMP_DIR, filename)

            # 简单的增量策略：如果本地已经有这个文件，且大小一样，就跳过
            # (如果需要确保绝对最新，可以注释掉下面这两行)
            if os.path.exists(local_path) and os.path.getsize(local_path) == obj.size:
                continue

            print(f"下载: {filename} ...")
            bucket.get_object_to_file(obj.key, local_path)
            download_count += 1

        if download_count == 0:
            print(Fore.GREEN + "本地文件已是最新，无需下载。")
        else:
            print(Fore.GREEN + f"成功下载 {download_count} 个新文件。")

    except Exception as e:
        print(Fore.RED + f"下载出错: {e}")
        return

    # 2. 合并与转换
    print(Fore.CYAN + "\n=== 2. 开始解压并合并为 market_history.csv ===")

    gz_files = sorted(glob(os.path.join(LOCAL_TEMP_DIR, "market_*.csv.gz")))
    if not gz_files:
        print(Fore.RED + "临时目录没有找到任何 .csv.gz 文件")
        return

    df_list = []
    print(f"正在处理 {len(gz_files)} 个压缩文件...")

    for f in gz_files:
        try:
            # Pandas 自动处理 gzip 解压
            df = pd.read_csv(f, compression='gzip')
            df_list.append(df)
        except Exception as e:
            print(Fore.YELLOW + f"警告: 文件 {f} 读取失败，已跳过。原因: {e}")

    if not df_list:
        return

    # 合并所有天的数据
    full_df = pd.concat(df_list, ignore_index=True)

    # 3. 数据清洗与还原 (适配旧脚本的关键步骤！)
    print("正在还原列名和格式...")

    # 将服务器的缩写列名 (t, i, l, a, b) 还原为全称
    # 如果有些老文件已经是全称了，rename 也是安全的
    full_df = full_df.rename(columns={
        't': 'timestamp',
        'i': 'item',
        'l': 'level',
        'a': 'ask',
        'b': 'bid'
    })

    # 确保生成 datetime 列 (你的训练脚本可能用到)
    if 'datetime' not in full_df.columns:
        full_df['datetime'] = pd.to_datetime(full_df['timestamp'], unit='s')

    # 按时间排序，保证数据连贯
    full_df = full_df.sort_values('timestamp')

    # 3.5 限制时间戳数量
    unique_timestamps = full_df['timestamp'].unique()
    total_timestamps = len(unique_timestamps)

    if MAX_TIMESTAMPS is not None and total_timestamps > MAX_TIMESTAMPS:
        print(f"检测到 {total_timestamps} 个时间戳，限制保留最近 {MAX_TIMESTAMPS} 个...")
        # 获取最近的 N 个时间戳
        keep_timestamps = unique_timestamps[-MAX_TIMESTAMPS:]
        full_df = full_df[full_df['timestamp'].isin(keep_timestamps)]
        total_timestamps = len(keep_timestamps)

    # 4. 保存为单一 CSV
    print(f"正在保存到 {FINAL_CSV_PATH} ...")
    full_df.to_csv(FINAL_CSV_PATH, index=False)

    file_size_mb = os.path.getsize(FINAL_CSV_PATH) / (1024 * 1024)
    print(Fore.GREEN + f"✅ 处理完成！")
    print(Fore.GREEN + f"有效时间戳数量: {total_timestamps}")
    print(Fore.GREEN + f"总行数: {len(full_df)}")
    print(Fore.GREEN + f"文件大小: {file_size_mb:.2f} MB")
    print(Fore.GREEN + f"路径: {os.path.abspath(FINAL_CSV_PATH)}")


if __name__ == "__main__":
    download_and_merge()
