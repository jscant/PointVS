"""Globally accessible config objects."""
import platform
import psutil

import torch

from point_vs import logging


LOG = logging.get_logger('PointVS')


# Use Cuda or MPS (Apple silicon) if available, CPU otherwise.
if torch.cuda.is_available(): # Cuda (best)
    DEVICE = torch.device('cuda')
    LOG.info('Using CUDA')
elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    LOG.info('Using MPS')
else: # Bog-standard CPU
    DEVICE = torch.device('cpu')
    LOG.info('Using CPU')

# Pytorch bug means setting this to more than 1 crashes on Winows/MacOS
NUM_WORKERS = min(4, psutil.cpu_count()) if platform.system() == 'Linux' else 0
