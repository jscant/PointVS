"""Globally accessible config objects."""
import platform
import psutil

import torch

# Use Cuda or MPS (Apple silicon) if available, CPU otherwise.
# Note: Apple Silicon (MPS) will be enabled once
# https://github.com/pytorch/pytorch/issues/77794 is solved.
if torch.cuda.is_available(): # Cuda (best)
    DEVICE = torch.device('cuda')
else: # Bog-standard CPU
    DEVICE = torch.device('cpu')

# Pytorch bug means setting this to more than 1 crashes on Winows/MacOS
NUM_WORKERS = min(4, psutil.cpu_count()) if platform.system() == 'Linux' else 0
