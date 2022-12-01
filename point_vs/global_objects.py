import platform
import psutil
import torch

global DEVICE
# Use Cuda or MPS (Apple silicon) if available, CPU otherwise.
if torch.cuda.is_available(): # Cuda (best)
    DEVICE = torch.device('cuda')
# M1/M2 Arm64 Apple Silicon Metal still very buggy with pygm, will uncomment when fixed.
# elif torch.backends.mps.is_available():
#    DEVICE = torch.device('mps')
else: # Bog-standard CPU
    DEVICE = torch.device('cpu')

# Pytorch bug means setting this to more than 1 crashes on Win/Linux
NUM_WORKERS = min(4, psutil.cpu_count()) if platform.system() == 'Linux' else 0