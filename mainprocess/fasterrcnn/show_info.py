import torch
TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
