import torch

def cpu() -> torch.device:  
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0) -> torch.device: 
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

def num_gpus() -> int:  
    """Return the number of available GPUs."""
    return torch.cuda.device_count()

def try_gpu(i=0) -> torch.device:  
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus() -> list[torch.device]:  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]
