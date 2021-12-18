import random
import numpy as np
import torch

# This files holds general utilities

GLOVE_PATH = "glove-twitter-200"
files_paths = {"train": "data/train.tagged", "test": "data/test.untagged", "dev": "data/dev.tagged"}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False