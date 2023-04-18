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

baseline = {
    "start": 1000,
    "defender": "none"
}
baseline_none = {
    "start": 1000,
    "defender": "none"
}
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

tr_mean = {
    "start": 50,
    "defender": "tr_mean"
}

median = {
    "start": 50,
    "defender": "median"
}

p_dense = {
    "start": 50,
    "defender": "p-dense"
}

p_cos = {
    "start": 50,
    "defender": "p-cosine"
}

p_merge = {
    "start": 50,
    "defender": "p-merge"
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

p_fang = {
    "start": 50,
    "defender": "p-fang"
}

p_trust = {
    "start": 50,
    "defender": "p-trust"
}

experiments = [baseline_dense,baseline_cos,baseline_fang,baseline_merge,baseline_none,baseline_fl_trust, attacked, fang, fl_trust, p_cos, p_dense, p_merge, p_trust, p_fang, np_cos, np_dense, np_merge]
poolings = [p_trust, p_fang, p_cos, p_dense, p_merge]

grad_ascent = ['grad_ascent']

att_modes = ["grad_ascent","mislead", "min_max", "label_flip", ]
targeted_att = ["scale"]

all_the_attack = ["grad_ascent","mislead", "min_max", "scale","label_flip"]

# all_the_attack = ["label_flip"]