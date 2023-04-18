import numpy as np
import random
import os
import torch

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch()
baseline_fang = {
    "start": 1000,
    "defender": "fang"
}
baseline_fl_trust = {
    "start": 1000,
    "defender": "fl_trust"
}
baseline_dense = {
    "start": 1000,
    "defender": "dense"
}
baseline_merge = {
    "start": 1000,
    "defender": "merge"
}

baseline_cos = {
    "start": 1000,
    "defender": "cos"
}
baseline = {
    "start": 1000,
    "defender": "none"
}

attacked = {
    "start": 50,
    "defender": "none"
}

fang = {
    "start": 50,
    "defender": "fang"
}

fl_trust = {
    "start": 50,
    "defender": "fl_trust"
}



np_dense = {
    "start": 50,
    "defender": "np-dense"
}

np_cos = {
    "start": 50,
    "defender": "np-cosine"
}

np_merge = {
    "start": 50,
    "defender": "np-merge"
}






grad_ascent = ['grad_ascent']
experiments = [baseline_fang,baseline_merge,baseline_dense,baseline_cos,baseline_fl_trust, attacked, fang, fl_trust,  np_cos, np_dense, np_merge]

att_modes = ["grad_ascent","mislead", "min_max", "label_flip", ]
targeted_att = ["scale"]

all_the_attack = ["mislead", "min_max", "label_flip","scale"]
# all_the_attack = ["label_flip"]

