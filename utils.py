import torch
import numpy as np
import random
import os

# Set the environment variable
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ':16:8'


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed_value)
