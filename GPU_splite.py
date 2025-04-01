import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
torch.cuda.set_per_process_memory_fraction(0.8, 0)
torch.cuda.empty_cache()
torch.cuda.memory_reserved()
torch.cuda.memory_allocated()
# 设置最大分割大小为 512MB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

torch.cuda.empty_cache()  # 清理GPU缓存
torch.cuda.memory_summary(device='cuda', abbreviated=True)  # 检查当前显存情况
