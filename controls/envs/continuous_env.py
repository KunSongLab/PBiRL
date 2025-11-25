import os
import sys
import copy
import monstac_api
import numpy as np
from gym import spaces
from pathlib import Path
from controls.envs.metagraph import MetaGraph
from controls.envs.commgraph import CommGraph
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MonstacEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}
    def __init__(self, scenario_name, device, random_seed=0, random_mean=0, random_std=0):
        self.rootpath = self.init_rootpath()
        self.scenario_name = scenario_name
        self.device = device
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
        self.actions = []
        self.meta_graph = None
        self.comm_graph = None
        self.step_count = 0
        self.init_agents()

    def step(self, action):
        for key, value in action.items():
            self.actions[key] = value

        done = self.base_env.step(self.actions)
        # observation, reward, done, info
        return self.get_observations(), \
               dict(zip(self.agent_ids, monstac_api.rl.get_rewards(self.base_env))), \
               dict(zip(self.agent_ids + ["__all__"], [done for _ in range(len(self.agent_ids) + 1)])), \
               dict(zip(self.agent_ids, [{} for _ in range(len(self.agent_ids) + 1)]))
        # return None, \
        #        dict(zip(self.agent_ids, monstac_api.rl.get_rewards(self.base_env))), \
        #        dict(zip(self.agent_ids + ["__top
        #        all__"], [done for _ in range(len(self.agent_ids) + 1)])), \
        #        dict(zip(self.agent_ids, [{} for _ in range(len(self.agent_ids) + 1)]))

    def reset(self):
        self.base_env.reset()
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
        self.comm_graph = CommGraph(self.base_env, self.meta_graph, self.device, 100, 0)
        self.comm_graph.build_graph()

        # initialize agents' observation spaces and action spaces
        for i, each in enumerate(self.meta_graph.subgraphs_pyg):
            self.agent_ids.append(i)
            self.actions.append(1)

            self.observation_spaces[i] = spaces.Dict({
                "own": spaces.Dict({
                    "x": spaces.Box(low=-1e-3, high=float("inf"), shape=each.x.shape, dtype=np.float32),
                    "edge_index": spaces.Box(low=-1e-3, high=1e7, shape=each.edge_index.shape, dtype=np.int64),
                    "edge_attr": spaces.Box(low=-1e-3, high=float("inf"), shape=each.edge_attr.shape, dtype=np.float32),
                    "center_id": spaces.Box(low=0, high=sys.maxsize, shape=[1], dtype=np.int64),
                    "batch": spaces.Box(low=-np.inf, high=np.inf, shape=each.y.shape, dtype=np.float32),
                })
            })
            # self.observation_spaces[i] = spaces.Box(low=-1e-3, high=float("inf"), shape=[each.x.shape[0]], dtype=np.float32)
            self.action_spaces[i] = spaces.Box(low=0, high=1, shape=[1], dtype=np.float32)
            # self.observation_space = self.observation_spaces[i]
            # self.action_space = self.action_spaces[i]

    def get_observation_space(self, agent):
        return self.observation_spaces[agent]

    def get_action_space(self, agent):
        return self.action_spaces[agent]

    def sample_actions(self):
        return dict(zip(self.agent_ids, np.random.rand(len(self.agent_ids))))

    def get_observations(self):
        temp_acc = monstac_api.rl.get_node_accumulation(self.base_env)
        temp_source = monstac_api.rl.get_node_source(self.base_env)
        temp_sink = monstac_api.rl.get_node_sink(self.base_env)
        temp_flow = monstac_api.rl.get_edge_flow(self.base_env)
        self.meta_graph.update_obs(temp_acc, temp_source, temp_sink, temp_flow)
        res = {}
        for i in range(len(self.agent_ids)):
            res[i] = {"own": {
                "x": copy.deepcopy(self.meta_graph.subgraphs_pyg[i].x),
                "edge_index": self.meta_graph.subgraphs_pyg[i].edge_index,
                "edge_attr": copy.deepcopy(self.meta_graph.subgraphs_pyg[i].edge_attr),
                "center_id": copy.deepcopy(self.meta_graph.subgraphs_pyg[i].center_id),
                "batch": self.meta_graph.subgraphs_pyg[i].batch
            }}
            # res[i] = self.meta_graph.subgraphs_pyg[i].x[:, 4].flatten()
        return res

    def get_episode_reward(self):
        remain_veh_t = monstac_api.general.get_running_vehicle_num(self.base_env) * monstac_api.general.get_simulation_seconds(self.base_env)
        finish_veh_t = monstac_api.general.get_total_running_time(self.base_env)
        inject_veh_num = monstac_api.general.get_injected_vehicle_num(self.base_env)
        return (remain_veh_t + finish_veh_t) / inject_veh_num