# MWI MarketSeer (Milky Way Idle 市场先知)

MWI MarketSeer 是一个专为 **Milky Way Idle** 游戏设计的市场数据分析与价格预测工具。它能够自动采集市场价格、进行深度数据清洗、训练机器学习模型（XGBoost），并提供下期涨幅预测及可视化展示。

---

## 🚀 核心功能

*   **自动化采集**：支持 24/7 实时爬取游戏市场 API，并同步至阿里云 OSS 存储。
*   **智能数据清洗**：
    *   **价差过滤**：自动剔除买卖价差过大（如“钓鱼价”）的异常记录。
    *   **波动过滤**：利用滚动中位数检测并移除价格“插针”现象。
    *   **稳定性筛选**：仅对持续有货的稳定商品进行建模。
*   **高维度特征工程**：
    *   引入滞后特征（Lag Features）、滚动波动率与动量。
    *   集成市场大盘收益率及 PCA 降维特征。
    *   采用正弦/余弦编码捕捉 24 小时周期性时间特征。
*   **个性化模型训练**：为数千种商品分别训练专属的 XGBoost 回归模型，支持留出法验证（Hold-out Validation）。
*   **精准预测与回测**：
    *   **潜力股排行榜**：实时显示预测涨幅最高的稳健商品。
    *   **误差监控**：展示模型在最近一期数据上的回测准确度。
    *   **专项监控**：针对 `milk` 类商品（牛奶及其衍生物）进行重点专项展示。
*   **趋势可视化**：自动绘制热门商品的价格走势图，支持按价格排序展示。

---

## 🛠️ 安装指南

1.  **克隆项目**
    ```bash
    git clone <project-url>
    cd MWI_MarketSeer
    ```

2.  **安装依赖**
    建议使用 Python 3.10+ 环境。
    ```bash
    pip install pandas numpy xgboost scikit-learn joblib tqdm colorama matplotlib schedule oss2 python-dotenv
    ```

3.  **环境配置 (可选)**
    如需使用云端存储功能，请在 `.env` 文件或环境变量中设置：
    ```env
    mwi_oss_ak=你的阿里云AccessKey
    mwi_oss_sk=你的阿里云SecretKey
    ```

---

## 📖 使用步骤

项目脚本按序号排列，建议依次运行：

1.  **数据积累 (`0_update_data.py`)**：
    运行爬虫。它每分钟会检查一次 API 并保存增量数据。建议运行至少 1 小时以上以获得初步训练数据。
    ```bash
    python 0_update_data.py
    ```

2.  **数据同步 (`1_download_data.py`)**：
    从 OSS 或本地目录汇总所有天的数据，生成统一的 `market_history.csv`。
    ```bash
    python 1_download_data.py
    ```

3.  **训练模型 (`2_train_model.py`)**：
    对市场中的稳定物品进行批量建模。完成后模型将保存在 `models/` 目录。
    ```bash
    python 2_train_model.py
    ```

4.  **执行预测 (`3_predict_prices.py`)**：
    查看最新的市场预测结果及排行榜。
    ```bash
    python 3_predict_prices.py
    ```

5.  **数据可视化 (`4_visualize_data.py`)**：
    查看关注商品（如 Milk 系列）的价格变动趋势。
    ```bash
    python 4_visualize_data.py
    ```

---

## 📂 项目结构

*   `0_update_data.py`: 数据爬虫及云端上传脚本。
*   `1_download_data.py`: 数据下载、合并与预处理脚本。
*   `2_train_model.py`: XGBoost 模型训练流水线（带进度条）。
*   `3_predict_prices.py`: 批量预测、回测验证及排行榜展示。
*   `4_visualize_data.py`: 基于 Matplotlib 的价格走势可视化。
*   `utils/utils.py`: 通用数据清洗、特征提取函数库。
*   `data/`: 存放原始历史数据。
*   `models/`: 存放训练好的 `.pkl` 模型文件。

---

## ⚠️ 免责声明

本工具仅供学习交流使用。预测结果基于历史数据统计规律，不代表市场真实成交保证。游戏市场有风险，投资（金币）需谨慎。
