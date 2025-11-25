import os
import gym
import time
import torch
import random
import monstac_api
import numpy as np
import matplotlib.pyplot as plt
from controls.envs.discrete_env import MonstacEnv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = MonstacEnv("RingFreeway", device, 0,0,0)


obs = env.reset()
done = {"__all__": False}
actions = {}
start = time.time()
while not done["__all__"]:
    for key, value in obs.items():
        # actions[key] = env.action_spaces[key].sample()
        actions[key] = 1
    # actions[0] = 1
    obs, reward, done, info = env.step(actions)

end = time.time()
print(end - start)
print(env.get_episode_reward())
print("The total demand is: ", monstac_api.general.get_injected_vehicle_num(env.base_env))
print("The running vehicle number is: ", monstac_api.general.get_running_vehicle_num(env.base_env))
print("The completed vehicle number is: ", monstac_api.general.get_completed_vehicle_num(env.base_env))
print("The split number is: ", monstac_api.general.get_split_num(env.base_env))
print("The total travel time (h): ", monstac_api.general.get_total_running_time(env.base_env)/3600)
print("The total truck travel distance is: ", monstac_api.general.get_total_truck_travel_distance(env.base_env)/1000)

