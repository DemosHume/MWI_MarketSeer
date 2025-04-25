# MWI_MarketSeer

## 项目概述

**MWI_MarketSeer** 是一个专注于市场数据预测的开源项目，旨在利用深度学习技术对市场数据进行分析和预测。
项目结合了 `BiLSTM_Attention` 和 `LSTM` 等先进的深度学习模型，对市场数据进行处理和学习，以预测商品价格的走势，帮助用户更好地把握市场动态。

### 数据来源

本项目的市场数据来源于 [MWIApi](https://github.com/holychikenz/MWIApi) 开源项目，该项目提供了最新的市场数据供分析和预测使用。
目前直接通过github仓库拉取数据，后续可能会考虑使用API接口获取数据。

## 功能特性

- **数据更新**：自动检测远程仓库的市场数据是否有更新，若有更新则自动下载最新的数据库文件，确保使用的数据是最新的。
- **数据预处理**：对市场数据进行清洗、异常值剔除和插值填充等操作，提高数据质量，为模型训练提供可靠的数据基础。
- **模型训练**：支持使用 `BiLSTM_Attention` 和 `LSTM` 模型进行训练，通过多轮迭代优化模型参数，提高预测的准确性。
- **商品价格预测**：使用训练好的模型对市场数据进行预测，输出可能上涨的商品及其涨幅预测，为用户提供决策参考。

## 目录结构

```
MWI_MarketSeer/
├── BiLSTM_Attention_Hike/
│   ├── data.py               # 数据加载和预处理
│   ├── main.py               # 项目主程序，控制训练和预测流程
│   ├── market.db             # 数据库文件，存储市场数据
│   ├── model.pth             # 训练好的模型参数
│   ├── model.py              # 定义 BiLSTM_Attention 模型结构
│   ├── predict.py            # 模型预测脚本
│   └── train.py              # 模型训练脚本
├── lstm/                     # 旧版本的模型和代码，后续版本应该会废弃
│   ├── bce_is_uper.py        # 使用 BCE 损失函数的预测脚本
│   └── lstm_pre.py           # LSTM 模型预测脚本
├── .gitignore                # Git 忽略文件列表
├── README.md                 # 项目说明文档
├── market.db                 # 市场数据库文件
├── market_data_info.json     # 存储市场数据的 commit hash 信息
├── pytorch_cuda_test.py      # 测试 PyTorch 是否支持 CUDA
├── requirements.txt          # 项目依赖库列表
├── utils.py                  # 通用工具函数，如数据查询、时间转换等
├── update_market.py.py       # 更新市场数据的脚本
└── 查看最新数据.ipynb          # Jupyter Notebook 查看最新数据
```

## 安装依赖

在运行项目之前，需要安装所需的 Python 依赖库。可以使用以下命令进行安装：

```bash
pip install -r requirements.txt
```

如有需要，cuda和pytorch另行安装

## 使用方法

### 1. 数据更新(可选)

运行 `update_market.py` ，检查市场数据是否有更新，并在有更新时下载最新的数据库文件：

```bash
python update_market.py
```

### 2. 模型训练

在 `BiLSTM_Attention_Hike` 文件夹中已经包含一份市场数据，为了保证模型的稳定性，该文件夹中的市场数据不会自动更新。

如果文件夹中的数据文件被删除，系统会自动从上级文件夹中拷贝最新的数据文件到 `BiLSTM_Attention_Hike`
文件夹中（后续可优化为在拷贝时附加最新时间戳，当前已标记为 TODO）。

可以通过修改 `BiLSTM_Attention_Hike/main.py` 中的 `MODE` 参数来选择训练模式：

- **新训练**：将 `MODE` 设置为 `'train'`，从头开始训练模型。
- **继续训练**：将 `MODE` 设置为 `'continue'`，加载已有模型参数并继续训练。

修改完成后，运行以下命令开始训练：

```bash
python BiLSTM_Attention_Hike/main.py

```

### 3. 模型预测

将 `BiLSTM_Attention_Hike/main.py` 中的 `MODE` 修改为 `'predict'`，使用训练好的模型进行预测：

```python
# 在 main.py 中修改 MODE
MODE = 'predict'

# 运行 main.py
python
BiLSTM_Attention_Hike / main.py
```

## 注意事项

- 运行数据更新脚本时，需要确保网络连接正常，并且可以访问 GitHub API。
- 如果使用 GPU 进行训练和预测，需要确保系统已经安装了 CUDA 和相应的 GPU 驱动。可以运行以下命令测试是否支持 CUDA：

```bash
python pytorch_cuda_test.py
```

- 本项目中的数据库文件 `market.db` 可能需要根据实际情况进行更新和维护。

## 贡献

欢迎对本项目进行贡献！如果你发现了 bug 或者有新的功能建议，可以通过以下方式参与：

- 提交 Issue 报告问题或提出功能需求。
- 提交 Pull Request 贡献代码。

## 许可证

本项目采用 **MIT 许可证**，详情请参阅 `LICENSE` 文件。