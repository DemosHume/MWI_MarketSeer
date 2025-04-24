import requests
from tqdm import tqdm

url = "https://github.com/holychikenz/MWIApi/raw/main/market.db"
proxy = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}
output_file = "market.db"

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

print("更新完成 ✅")
