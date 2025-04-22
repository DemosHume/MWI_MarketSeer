import torch

# 检查是否有 GPU 可用
if torch.cuda.is_available():
    print("CUDA is available. GPU is enabled.")
    # 输出 GPU 的名称
    print("GPU Name:", torch.cuda.get_device_name(0))
    # 新建一个output.txt
    with open("output.txt", "w") as f:
        f.write("CUDA is available. GPU is enabled.\n")
        f.write("GPU Name: " + torch.cuda.get_device_name(0) + "\n")
else:
    print("CUDA is not available. Using CPU.")
