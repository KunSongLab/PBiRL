import os
import json
import sys
import copy
import torch
import monstac_api
import numpy as np
from gymnasium import spaces
from pathlib import Path
from collections import deque
from controls.envs.metagraph import MetaGraph
from controls.envs.commgraph import CommGraph
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MonstacEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}
    def __init__(self, scenario_name, device, random_seed, random_mean, random_std):
        self.rootpath = self.init_rootpath()
        self.scenario_name = scenario_name
        self.device = device
        self.window_size = 3
        self.random_seed = random_seed
        self.random_mean = random_mean
        self.random_std = random_std
        self.config_filepath = os.path.join(self.rootpath, "scenarios", self.scenario_name + ".json")
        self.base_env = monstac_api.general.Monstac(self.config_filepath, random_seed, random_mean, random_std)
        self.agent_ids = []
        self.observation_spaces = {}
        self.action_spaces = {}
        self.observation_space = None
        self.action_space = None
        self.observations = {}
        self.reward_baselines = {}
        self.observations_his = {}
        self.actions = []
        self.meta_graph = None
        self.comm_graph = None
        self.step_count = 0
        self.init_agents()

        # TODO
        # agent_ids_dict ={}
        # temp = []
        # print(self.meta_graph.controllable_agent_indices)
        # for i, each in enumerate(self.meta_graph.controllable_agent_indices):
        #     print(i, ", ", monstac_api.rl.get_init_node_observations(self.base_env).location[each])
        #     agent_ids_dict[f"{i}"] = monstac_api.rl.get_init_node_observations(self.base_env).location[each]
        #     temp.append(monstac_api.rl.get_init_node_observations(self.base_env).location[each])
        # print(np.array(temp))
        # with open("agent_ids_dict.json", "w") as f:
        #     json.dump(agent_ids_dict, f)

        self.get_reward_baseline()

    def step(self, action):
        for key, value in action.items():
            if value == 0:
                self.actions[key] = max(min(self.actions[key] - 1, 1), 0.1)
            elif value == 2:
                self.actions[key] = max(min(self.actions[key] + 1, 1), 0.1)

        done = self.base_env.step(self.actions)
        # observation, reward, done, info
        temp_reward = dict(zip(self.agent_ids, monstac_api.rl.get_rewards(self.base_env)[1]))
        temp_obs = self.get_observations()
        # print("reward_baselines:", self.reward_baselines)

        # print("step_count:", self.step_count)
        for agent_id in self.agent_ids:
            # print("speed : ", temp_reward[agent_id],"speed base : ", self.reward_baselines[agent_id][self.step_count])
            temp_reward[agent_id] -= self.reward_baselines[agent_id][self.step_count]
            # print("reward_time", temp_reward[agent_id])
            # print("len:", len(self.reward_baselines[agent_id]))
            # temp_reward[agent_id] = -(temp_obs[0]["own"]["x"][0, 4] - 0.6) ** 2

        self.step_count += 1
        return temp_obs, \
               temp_reward, \
               dict(zip(self.agent_ids + ["__all__"], [done for _ in range(len(self.agent_ids) + 1)])), \
               dict(zip(self.agent_ids, [{} for _ in range(len(self.agent_ids) + 1)]))

    def reset(self):
        self.step_count = 0
        self.base_env.reset()
        for i in range(len(self.agent_ids)):
            self.observations_his[i].clear()

        return self.get_observations()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def init_rootpath(self):
        p = Path(__file__).absolute()
        pparts = list(p.parts)
        count = -1
        for i, each in enumerate(pparts):
            if each == "Monstac":
                count = i
                break
        rootpath = os.path.join(*pparts[:count + 1])
        os.chdir(rootpath)
        return rootpath

    def init_agents(self):
        # initialize two-layer graph
        self.meta_graph = MetaGraph(self.base_env, self.device, 100, 0)
        self.meta_graph.build_graph()
        # self.meta_graph.show_subgraphs()
        self.comm_graph = CommGraph(self.base_env, self.meta_graph, self.device, 100, 0)
        self.comm_graph.build_graph()
        # initialize agents' observation spaces and action spaces
        for i, each in enumerate(self.meta_graph.subgraphs_pyg):
            self.agent_ids.append(i)
            self.actions.append(1)
            self.reward_baselines[i] = []
            self.observation_spaces[i] = spaces.Dict({
                "own": spaces.Dict({
                    "x": spaces.Box(low=-1e-3, high=float("inf"), shape=each.x.shape, dtype=np.float32),
                    "edge_index": spaces.Box(low=-1e-3, high=1e7, shape=each.edge_index.shape, dtype=np.int64),
                    "edge_attr": spaces.Box(low=-1e-3, high=float("inf"), shape=each.edge_attr.shape, dtype=np.float32),
                    "center_id": spaces.Box(low=0, high=sys.maxsize, shape=[1], dtype=np.int64),
                    "batch": spaces.Box(low=-np.inf, high=np.inf, shape=each.y.shape, dtype=np.float32),
                }),
                # "history": spaces.Dict({
                #     "x": spaces.Box(low=-1e-3, high=float("inf"),
                #                     shape=list(each.x.shape) + [self.window_size],
                #                     dtype=np.float32),
                #     "edge_index": spaces.Box(low=-1e-3, high=1e7,
                #                              shape=list(each.edge_index.shape) + [self.window_size],
                #                              dtype=np.int64),
                #     "edge_attr": spaces.Box(low=-1e-3, high=float("inf"),
                #                             shape=list(each.edge_attr.shape) + [self.window_size],
                #                             dtype=np.float32),
                #     "batch": spaces.Box(low=-np.inf, high=np.inf,
                #                         shape=list(each.y.shape) + [self.window_size],
                #                         dtype=np.float32),
                # }),
            })
            self.action_spaces[i] = spaces.Discrete(3) # -0.05, 0, 0.05
            self.observations_his[i] = deque(maxlen=self.window_size)

    def get_observation_space(self, agent):
        return self.observation_spaces[agent]

    def get_action_space(self, agent):
        return self.action_spaces[agent]

    def sample_actions(self):
        return dict(zip(self.agent_ids, np.random.rand(len(self.agent_ids))))

    def get_observations(self):
        res = {}
        own_obs = self.get_own_observations()
        # self.update_his_observation(own_obs)
        # his_obs = self.convert_his_observation()

        for i in range(len(self.agent_ids)):
            # res[i] = {"own": own_obs[i],
            #           "history": his_obs[i]}
            res[i] = {"own": own_obs[i]}
        return res

    def get_own_observations(self):
        temp_acc = monstac_api.rl.get_node_accumulation(self.base_env)
        temp_source = monstac_api.rl.get_node_source(self.base_env)
        temp_sink = monstac_api.rl.get_node_sink(self.base_env)
        temp_flow = monstac_api.rl.get_edge_flow(self.base_env)
        self.meta_graph.update_obs(temp_acc, temp_source, temp_sink, temp_flow)
        res = {}
        for i in range(len(self.agent_ids)):
            res[i] = {
                "x": copy.deepcopy(self.meta_graph.subgraphs_pyg[i].x),
                "edge_index": self.meta_graph.subgraphs_pyg[i].edge_index,
                "edge_attr": copy.deepcopy(self.meta_graph.subgraphs_pyg[i].edge_attr),
                "center_id": copy.deepcopy(self.meta_graph.subgraphs_pyg[i].center_id),
                "batch": self.meta_graph.subgraphs_pyg[i].batch
            }
        return res

    def update_his_observation(self, obs):
        for i in range(len(self.agent_ids)):
            if len(self.observations_his[i]) == 0:
                for _ in range(self.window_size - 1):
                    self.observations_his[i].append(obs[i])
            self.observations_his[i].append(obs[i])

    def convert_his_observation(self):
        res = {}
        for i in range(len(self.agent_ids)):
            res[i] = {}
            res[i]["x"] = torch.stack([each["x"] for each in self.observations_his[i]], dim=-1)
            res[i]["edge_index"] = torch.stack([each["edge_index"] for each in self.observations_his[i]], dim=-1)
            res[i]["edge_attr"] = torch.stack([each["edge_attr"] for each in self.observations_his[i]], dim=-1)
            res[i]["batch"] = torch.stack([each["batch"] for each in self.observations_his[i]], dim=-1)
        return res

    def get_episode_reward(self):
        # remain_veh_t = monstac_api.general.get_running_vehicle_num(self.base_env) * monstac_api.general.get_simulation_seconds(self.base_env)
        # finish_veh_t = monstac_api.general.get_total_running_time(self.base_env)
        # inject_veh_num = monstac_api.general.get_injected_vehicle_num(self.base_env)
        # return (remain_veh_t + finish_veh_t) / inject_veh_num
        total_veh_t = monstac_api.general.get_total_running_time(self.base_env)
        return total_veh_t

    def get_reward_baseline(self):
        temp_action = {}
        for agent_id in self.agent_ids:
                temp_action[agent_id] = 1

        obs = self.reset()
        done = False
        while not done:
            done = self.base_env.step(self.actions)
            temp_reward = dict(zip(self.agent_ids, monstac_api.rl.get_rewards(self.base_env)[1]))
            for agent_id in self.agent_ids:
                self.reward_baselines[agent_id].append(temp_reward[agent_id])
                # print("reward_baselines[agent_id]_len:", len(self.reward_baselines[agent_id]))
                # print("temp_reward[agent_id]:", temp_reward[agent_id])
        return self.reset()