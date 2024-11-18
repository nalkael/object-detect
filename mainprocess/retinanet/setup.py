import torch, detectron2

# Basic setup and import
from detectron2.utils.logger import setup_logger


TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2: ", detectron2.__version__)