import os
import torch

def set_cpu_threads(n=None):
    if n is None:
        n = os.cpu_count() or 8
    torch.set_num_threads(n)
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
