import os
import oss2
from dotenv import load_dotenv
from colorama import init, Fore

# 初始化
init(autoreset=True)
load_dotenv()  # 加载本地 .env 文件

# === 配置 ===
OSS_BUCKET_NAME = 'milky-way-idle-oss'
OSS_ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'  # 你的 Endpoint
OSS_OBJECT_KEY = 'milkyway/market_history.csv'  # 云端路径
LOCAL_SAVE_PATH = 'data/market_history.csv'  # 本地保存路径


def download_from_oss():
    ak = os.getenv('demos_oss_ak')
    sk = os.getenv('demos_oss_sk')

    if not ak or not sk:
        print(Fore.RED + "错误: 未找到环境变量 demos_oss_ak 或 demos_oss_sk")
        return

    # 确保本地目录存在
    os.makedirs(os.path.dirname(LOCAL_SAVE_PATH), exist_ok=True)

    print(Fore.CYAN + f"正在连接 OSS: {OSS_BUCKET_NAME}...")
    try:
        auth = oss2.Auth(ak, sk)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)

        # 检查文件是否存在
        if not bucket.object_exists(OSS_OBJECT_KEY):
            print(Fore.RED + "错误: 云端文件不存在！请确认服务器脚本是否已运行并上传。")
            return

        # 下载
        print(Fore.YELLOW + "开始下载数据...")
        bucket.get_object_to_file(OSS_OBJECT_KEY, LOCAL_SAVE_PATH)

        file_size = os.path.getsize(LOCAL_SAVE_PATH) / 1024
        print(Fore.GREEN + f"下载成功！文件保存在: {LOCAL_SAVE_PATH}")
        print(Fore.GREEN + f"文件大小: {file_size:.2f} KB")

    except Exception as e:
        print(Fore.RED + f"下载失败: {e}")


if __name__ == "__main__":
    download_from_oss()
