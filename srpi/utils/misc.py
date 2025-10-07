import numpy as np

def set_seed(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
