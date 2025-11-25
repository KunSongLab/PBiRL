import os
import gym
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from controls.envs.discrete_env import MonstacEnv
from controls.models.dqn_priority.dqn_priority import PriorityDQN

parser = argparse.ArgumentParser(description='Finetune hyperparameters')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha')
parser.add_argument('--beta', type=float, default=0.4, help='beta')
parser.add_argument('--frame', type=int, default=10000, help='frame')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = MonstacEnv("RingFreeway", device, 0,0,0)
config = {
    "name": "RingFreeway",
    "train": {
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 0.3,
        "epsilon_end": 0.05,
        "epsilon_decay": 500,
        "epoch": 500,
        "batch_size": 128,
    },
    "model": {
        "target_update_freq": 5,
        0: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        1: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        2: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        3: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        4: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        5: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        6: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        8: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        9: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        10: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        11: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        12: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        13: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        14: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        15: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        16: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        17: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        18: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        19: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        20: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        21: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        22: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        23: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        24: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        25: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        26: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        27: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        28: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        29: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        30: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        31: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        32: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        33: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        34: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        35: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        36: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        37: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        38: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        39: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        40: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        41: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        42: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        43: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        44: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        45: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        46: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        47: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        48: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        49: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        50: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        51: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        52: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        53: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        54: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        55: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        56: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        57: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        58: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        59: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        60: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        61: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        62: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        63: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        64: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        65: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        66: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        67: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
        68: {"input_dims": 15, "hidden_dims": 50, "output_dims": 3},
    },
    "memory": {
        "max_memory_len": 10000,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 10000
    }
}

config["memory"]["alpha"] = args.alpha
config["memory"]["beta_start"] = args.beta
config["memory"]["max_memory_len"] = args.frame
config["memory"]["beta_frames"] = args.frame

obs = env.reset()
dqn = PriorityDQN(config, device, [10, 6, 32, 14, 2, 40, 36, 22, 24, 28, 0])
# dqn = PriorityDQN(config, device, [15, 16, 9, 10, 54, 55, 21, 22, 59, 60, 3, 4, 62, 63, 57, 58, 35, 36, 38, 39])
res = dqn.train(env)
dqn.save_results(res, os.path.join(dqn.savepath, "stat1.csv"))
