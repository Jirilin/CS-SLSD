import random
import os
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Sets global random seeds across Python, NumPy, and PyTorch for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def choose_device(requested_device: str = "auto") -> torch.device:
    """
    Selects the compute device (MPS for Mac M-series GPUs, CUDA for NVIDIA GPUs, or CPU).
    """
    if requested_device is None or requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(requested_device)