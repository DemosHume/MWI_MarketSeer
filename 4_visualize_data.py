import pandas as pd
import matplotlib.pyplot as plt
import os
from colorama import init, Fore
from utils.utils import load_clean_data

init(autoreset=True)

# === 配置 ===
DATA_PATH = 'data/market_history.csv'

def visualize():
    # 1. 加载数据
    print(Fore.CYAN + "正在加载并清洗数据...")
    pivot_df = load_clean_data(DATA_PATH)
    
    if pivot_df is None or pivot_df.empty:
        print(Fore.RED + "没有可用的数据，请确保已运行爬虫积累数据。")
        return

    # 将索引转换为 datetime 格式以便绘图
    # pivot_df 的索引是 timestamp (int)
    try:
        pivot_df.index = pd.to_datetime(pivot_df.index, unit='s')
    except Exception as e:
        print(Fore.YELLOW + f"时间戳转换失败，将直接使用原始索引: {e}")

    # 2. 筛选商品名为 milk 或以 _milk 结尾的物品
    # 商品 ID 格式为 {item}_lv{level}
    all_selected = [col for col in pivot_df.columns if col.startswith('milk_lv') or '_milk_lv' in col]
    
    if not all_selected:
        print(Fore.RED + "未找到任何 milk 类商品，请检查数据。")
        return

    # === 新增：按平均价格从高到低排序 ===
    # 计算每个物品的平均价格
    avg_prices = pivot_df[all_selected].mean().sort_values(ascending=False)
    selected_items = avg_prices.index.tolist()
    
    print(Fore.GREEN + f"成功加载数据！共 {len(pivot_df.columns)} 个稳定物品。")
    print(Fore.YELLOW + f"已自动筛选并按价格排序 {len(selected_items)} 个 'milk' 类物品。")

    # 3. 绘图限制
    MAX_PLOT = 30
    if len(selected_items) > MAX_PLOT:
        print(Fore.YELLOW + f"警告: 匹配物品过多 ({len(selected_items)})，仅绘制价格最高的前 {MAX_PLOT} 个。")
        selected_items = selected_items[:MAX_PLOT]

    # 4. 绘图
    print(Fore.CYAN + f"\n正在准备图表，包含 {len(selected_items)} 个物品...")
    
    plt.figure(figsize=(12, 7))
    for item in selected_items:
        # 剔除可能存在的 NaN 以防止断线
        series = pivot_df[item].dropna()
        if not series.empty:
            plt.plot(series.index, series.values, marker='.', markersize=4, label=item)

    plt.title('MWI Market - Item Price Trends')
    plt.xlabel('Time')
    plt.ylabel('Price (Ask)')
    
    # 图例放在右侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 尝试显示
    print(Fore.GREEN + "图表已生成。")
    try:
        # 尝试非阻塞式显示，或者提示用户
        print("正在尝试打开可视化窗口...")
        plt.show()
    except Exception as e:
        save_path = 'price_trends.png'
        plt.savefig(save_path)
        print(Fore.YELLOW + f"无法打开交互式窗口 (可能是远程环境)，图表已保存至: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    visualize()
