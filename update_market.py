"""
更新市场数据脚本

此脚本用于调用 `refresh_market_data` 函数以刷新市场数据。
"""

from utils import refresh_market_data

# 主程序入口
if __name__ == '__main__':
    # 调用刷新市场数据的函数
    refresh_market_data()