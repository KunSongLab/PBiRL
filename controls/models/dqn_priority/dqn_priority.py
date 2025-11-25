import os.path
import time
import math
import torch
import itertools
import numpy as np
import pandas as pd
from torch import nn
import datetime as dt
import matplotlib.pyplot as plt
from controls.models.dqn_priority.models import DQNModel
from controls.models.dqn_priority.replay_memory import MAPrioritizedReplayMemory, Transition


class PriorityDQN(object):
    def __init__(self, config, device, agent_ids):
        self.name = config["name"]
        self.lr = config["train"]["learning_rate"]
        self.gamma = config["train"]["gamma"]
        self.epsilon_start = config["train"]["epsilon_start"]
        self.epsilon_end = config["train"]["epsilon_end"]
        self.epsilon_decay = config["train"]["epsilon_decay"]
        self.epoch = config["train"]["epoch"]
        self.batch_size = config["train"]["batch_size"]
        self.max_memory_len = config["memory"]["max_memory_len"]
        self.alpha = config["memory"]["alpha"]
        self.beta_start = config["memory"]["beta_start"]
        self.beta_frames = config["memory"]["beta_frames"]
        self.num_outputs = {}
        self.target_update_freq = config["model"]["target_update_freq"]
        self.agent_ids = agent_ids
        self.device = device
        self.featurizer = None
        self.scaler = None
        self.q_networks = {}
        self.target_networks = {}
        self.optims = {}
        self.init_models(config)
        self.replay_memory = MAPrioritizedReplayMemory(
            self.agent_ids, self.device, self.max_memory_len,
            self.alpha, self.beta_start, self.beta_frames)
        self.criterion = nn.SmoothL1Loss()
        cur_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.savepath = os.path.join("controls/run", self.name, "outputs", cur_time)
        os.makedirs(self.savepath)
        print(cur_time)

    def init_models(self, config):
        for agent_id in self.agent_ids:
            self.num_outputs[agent_id] = config["model"][agent_id]["output_dims"]
            self.q_networks[agent_id] = DQNModel(config["model"][agent_id]["input_dims"],
                                                 config["model"][agent_id]["hidden_dims"],
                                                 config["model"][agent_id]["output_dims"]).to(self.device)
            self.target_networks[agent_id] = DQNModel(config["model"][agent_id]["input_dims"],
                                                      config["model"][agent_id]["hidden_dims"],
                                                      config["model"][agent_id]["output_dims"]).to(self.device)
            if config["model"][agent_id].__contains__("init_model") and config["model"][agent_id]["init_model"] != None:
                self.q_networks[agent_id].load_state_dict(torch.load(config["model"][agent_id]["init_model"]))
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
            self.optims[agent_id]= torch.optim.Adam(self.q_networks[agent_id].parameters(), lr=self.lr)

    def train(self, env):
        max_travel_time = 1e5
        training_record_dict = self.init_training_dict()
        for i in range(self.epoch):
            obs = env.reset()
            episode_record_dict = self.init_episode_dict()
            action_list = dict(zip(self.agent_ids, [[] for _ in range(len(self.agent_ids))]))
            for j in itertools.count():
                multi_actions = {}
                multi_actions_memory = {}
                for agent_id in self.agent_ids:
                    single_action_value = self.q_networks[agent_id]([obs[agent_id]])
                    single_action = self.select_action(single_action_value, agent_id, i)
                    multi_actions[agent_id] = single_action
                    multi_actions_memory[agent_id] = torch.from_numpy(np.array([[single_action]], dtype=np.int64)).to(self.device)
                    action_list[agent_id].append(single_action)
                next_obs, reward, done, _ = env.step(multi_actions)
                # print("============")
                # print(len(reward))
                # print(reward)
                # print("=============")

                for agent_id in self.agent_ids:
                    reward[agent_id] = torch.from_numpy(np.array([[reward[agent_id]]], dtype=np.float32)).to(self.device)
                    # print("agent_id:",agent_id)
                    # print("reward:", reward[agent_id])
                if done["__all__"]:
                    next_obs = dict(zip(self.agent_ids, [None for _ in range(len(self.agent_ids))]))
                self.replay_memory.push(obs, multi_actions_memory, next_obs, reward)

                loss = self.update_model()
                episode_record_dict = self.episode_record(episode_record_dict, obs, multi_actions, reward, next_obs, loss)
                obs = next_obs

                if done["__all__"]:
                    break
            if i % self.target_update_freq == 0:
                for agent_id in self.agent_ids:
                    self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())

            training_record_dict = self.training_record(training_record_dict, episode_record_dict, env)
            print("Iteration: ", i, "Average reward: ", training_record_dict["reward"][-1], end="\t")
            print("Total Travel time", env.get_episode_reward())

            if (env.get_episode_reward() < max_travel_time):
                max_travel_time = env.get_episode_reward()
                save_action_list = {}
                for agent_id in self.agent_ids:
                    torch.save(self.q_networks[agent_id],
                               os.path.join(self.savepath, "agent_%d_model_%05d.pt" % (agent_id, i)))
                    save_action_list[str(agent_id)] = np.array(action_list[agent_id])
                np.savez(os.path.join(self.savepath, "action_%05d.npz" % i), **save_action_list)
        return training_record_dict

    def test(self, env, model_path):
        obs = env.reset()
        actions = []
        for agent_id in self.agent_ids:
            self.q_networks[agent_id] = torch.load(model_path)
        for i in itertools.count():
            multi_actions = {}
            for agent_id in self.agent_ids:
                action_value = self.q_networks[agent_id]([obs[agent_id]])
                action = torch.argmax(action_value).numpy()
                multi_actions[agent_id] = action
            obs, reward, done, _ = env.step(multi_actions)
            actions.append(env.actions[0])
            if done["__all__"]:
                break
        print(actions)
        plt.plot(actions)
        plt.show()
        print(env.get_episode_reward())

    def select_action(self, action_value, agent_id, iteration):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * iteration / self.epsilon_decay)
        if np.random.uniform() > eps_threshold:
            with torch.no_grad():
                return torch.argmax(action_value).cpu().numpy()
        else:
            return np.random.randint(0, self.num_outputs[agent_id], dtype=np.int32)

    def update_model(self):
        if len(self.replay_memory) < self.batch_size:
            return dict(zip(self.agent_ids, [0.0 for _ in range(len(self.agent_ids))]))

        transitions, indices, weights = self.replay_memory.sample(self.batch_size)
        losses = {}
        for agent_id in self.agent_ids:
            batch = Transition(*zip(*transitions[agent_id]))
            # mask out the non-final state
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_obs)), dtype=torch.bool)
            non_final_next_obs = [s for s in batch.next_obs if s is not None]
            obs_batch = list(batch.obs)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            q_values = self.q_networks[agent_id](obs_batch).gather(1, action_batch)
            next_obs_value = torch.zeros(self.batch_size, device=self.device)
            next_obs_value[non_final_mask] = self.target_networks[agent_id](non_final_next_obs).max(1)[0].detach()
            expected_q_values = (torch.unsqueeze(next_obs_value, 1) * self.gamma) + reward_batch

            diff = (expected_q_values - q_values)
            self.replay_memory.update_priority(agent_id, indices[agent_id],
                                               diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = torch.square(diff).squeeze() * weights[agent_id]
            loss = loss.mean()
            self.optims[agent_id].zero_grad()
            loss.backward()
            # for param in self.q_network.parameters():
            #     param.grad.clamp_(-1, 1) # clip the gradient.
            self.optims[agent_id].step()
            losses[agent_id] = loss
        return losses

    def init_episode_dict(self):
        return {"obs": [], "action": [], "reward": [], "next_obs": [], "loss": []}

    def episode_record(self, res, obs, action, reward, next_obs, loss):
        res["obs"].append(obs)
        res["action"].append(action)
        res["reward"].append(reward)
        res["next_obs"].append(next_obs)
        res["loss"].append(loss)
        return res

    def init_training_dict(self):
        return {"reward": [], "loss": [], "horizon_length": [], "ttt": []}

    def training_record(self, res, episode_records, env):
        mean_reward = 0
        mean_loss = 0
        for agent_id in self.agent_ids:
            for i in range(len(episode_records["loss"])):
                mean_reward += episode_records["reward"][i][agent_id]
                mean_loss += episode_records["loss"][i][agent_id]
        mean_reward = (mean_reward / len(self.agent_ids) /len(episode_records["loss"])).detach().cpu().numpy()[0, 0]
        # mean_reward = (mean_reward / len(self.agent_ids)).detach().cpu().numpy()[0, 0]
        mean_loss = (mean_loss / len(self.agent_ids) / len(episode_records["loss"])).detach().cpu().numpy()
        # print("reward:", mean_reward)
        res["reward"].append(mean_reward)
        res["loss"].append(mean_loss)
        res["horizon_length"].append(len(episode_records["obs"]))
        res["ttt"].append(env.get_episode_reward())
        return res

    def save_results(self, training_record_dict, savepath):
        training_record_dict = pd.DataFrame(training_record_dict)
        training_record_dict.to_csv(savepath, index=False)
