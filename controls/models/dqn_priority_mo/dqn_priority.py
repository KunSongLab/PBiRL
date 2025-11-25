import os.path
import math
import torch
import itertools
import numpy as np
import pandas as pd
from torch import nn
import datetime as dt
import matplotlib.pyplot as plt
from controls.models.dqn_priority_mo.models import DQNModel
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
        self.criterion = nn.SmoothL1Loss(reduction="none")
        cur_time = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.savepath = os.path.join("controls/run", self.name, "outputs", cur_time)
        os.makedirs(self.savepath)
        print(cur_time)
        self.controllable_agent_num = 46  # 46
        self.total_actions = np.empty((0, self.controllable_agent_num))
        # init
        self.speed_running_mean = 0.0
        self.toll_running_mean = 0.0
        self.speed_running_var = 1.0
        self.toll_running_var = 1.0
        self.beta = 0.99
        self.eps = 1e-8

    def init_models(self, config):
        for agent_id in self.agent_ids:
            self.num_outputs[agent_id] = config["model"][agent_id]["output_dims"]
            self.q_networks[agent_id] = DQNModel(config["model"][agent_id]["input_dims"],
                                                 config["model"][agent_id]["hidden_dims"],
                                                 config["model"][agent_id]["output_dims"]).to(self.device)
            self.target_networks[agent_id] = DQNModel(config["model"][agent_id]["input_dims"],
                                                      config["model"][agent_id]["hidden_dims"],
                                                      config["model"][agent_id]["output_dims"]).to(self.device)
            if config["model"][agent_id].__contains__("init_model") and config["model"][agent_id][
                "init_model"] is not None:
                self.q_networks[agent_id].load_state_dict(
                    torch.load(config["model"][agent_id]["init_model"]).state_dict())
            self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
            self.optims[agent_id] = torch.optim.Adam(self.q_networks[agent_id].parameters(), lr=self.lr)

    def controlled_agent_num_optim(self, window_size: int = 7200, metering_rate_threshold: float = 0.9):
        remove_agent_ids = []
        if self.total_actions.shape[0] >= window_size:
            for each in range(self.total_actions.shape[1]):
                if all(self.total_actions[-window_size:, each] >= metering_rate_threshold) and each in self.agent_ids:
                    self.agent_ids.remove(each)
                    remove_agent_ids.append(each)
            self.total_actions = np.delete(self.total_actions, [0], axis=0)
            return remove_agent_ids

    def train(self, env):
        training_record_dict = self.init_training_dict()
        step_count = 0
        for i in range(self.epoch):
            print("Agent ID List:", self.agent_ids)
            obs = env.reset()
            episode_record_dict = self.init_episode_dict()
            action_list = dict(zip(self.agent_ids, [[] for _ in range(len(self.agent_ids))]))
            for j in itertools.count():  # (while True:)
                multi_actions = {}
                multi_actions_memory = {}
                for agent_id in self.agent_ids:
                    single_action_value = self.q_networks[agent_id]([obs[agent_id]])
                    single_action = self.select_action(single_action_value, i)
                    multi_actions[agent_id] = single_action
                    multi_actions_memory[agent_id] = torch.from_numpy(np.array([[single_action]], dtype=np.int64)).to(
                        self.device)
                    action_list[agent_id].append(single_action)
                next_obs, reward, done, _ = env.step(multi_actions)
                # self.total_actions = np.vstack([self.total_actions, np.array(env.actions)])
                # print(env.actions[3])
                # print(self.total_actions[:, 3])
                # self.controlled_agent_num_optim(window_size=3600,  metering_rate_threshold=0.9)
                # if remove_agent_ids:
                #     for remove_agent_id in remove_agent_ids:
                #         env.actions[remove_agent_id] = 1

                for agent_id in self.agent_ids:
                    # 假设 reward[agent_id] 为字典，包含 "speed" 与 "toll"
                    reward[agent_id]["speed"] = torch.from_numpy(
                        np.array([[reward[agent_id]["speed"]]], dtype=np.float32)).to(self.device)
                    reward[agent_id]["toll"] = torch.from_numpy(
                        np.array([[reward[agent_id]["toll"]]], dtype=np.float32)).to(self.device)

                if done["__all__"]:
                    next_obs = dict(zip(self.agent_ids, [None for _ in range(len(self.agent_ids))]))
                self.replay_memory.push(obs, multi_actions_memory, next_obs, reward)

                loss = self.update_model(step_count)
                step_count += 1
                # print(loss)
                episode_record_dict = self.episode_record(episode_record_dict, obs, multi_actions, reward, next_obs,
                                                          loss)
                obs = next_obs

                if done["__all__"]:
                    break
            if i % self.target_update_freq == 0:
                for agent_id in self.agent_ids:
                    self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())
            # print(episode_record_dict)
            training_record_dict = self.training_record(training_record_dict, episode_record_dict, env)
            print("Iteration: ", i, end="\t")
            print("Average reward speed (km/h): ", training_record_dict["reward_speed"][-1], end="\t")
            print("Average reward toll (¥): ", training_record_dict["reward_toll"][-1], end="\t")
            print("Total travel time (h): ", env.get_episode_reward_time() / 3600, end="\t")
            print("Total toll revenue (¥): ", env.get_episode_reward_toll() / 1000)
            # 若当前总旅行时间更小，则保存模型与动作记录
            if (True):
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
        print(env.get_episode_reward_speed())

    def execute_action(self, env, actions):
        obs = env.reset()
        idx = 0
        while True:
            multi_actions = {}
            for key in actions.keys():
                multi_actions[int(key)] = actions[key][idx]
            obs, reward, done, _ = env.step(multi_actions)
            idx += 1
            if done["__all__"]:
                break

    def select_action(self, action_value, iteration):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * iteration / self.epsilon_decay)
        q_speed, q_toll, _ = action_value
        q_speed = q_speed.squeeze().cpu().detach().numpy()  # shape: (num_actions,)
        q_toll = q_toll.squeeze().cpu().detach().numpy()
        if np.random.uniform() > eps_threshold:
            with torch.no_grad():
                weights_pool = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
                normalized_q_speed = (q_speed - np.min(q_speed)) / (np.max(q_speed) - np.min(q_speed))
                normalized_q_toll = (q_toll - np.min(q_toll)) / (np.max(q_toll) - np.min(q_toll))
                pareto_indices = self.pareto_front_indices(q_speed, q_toll)
                if len(pareto_indices) > 0:
                    idx = iteration % len(weights_pool)
                    w = weights_pool[idx]
                    sums = [normalized_q_speed[i] * (1 - w) + normalized_q_toll[i] * w for i in pareto_indices]
                    chosen_index = pareto_indices[np.argmax(sums)]
                    return chosen_index
                    # return np.random.choice(pareto_indices)
                else:
                    return int(np.argmax(normalized_q_speed + normalized_q_toll))
        else:
            # return np.argmax(q_speed)
            return np.random.randint(0, self.num_outputs[agent_id], dtype=np.int32)

    def update_model(self, step_count):
        losses = {}
        if len(self.replay_memory) < self.batch_size:
            for agent_id in self.agent_ids:
                losses[agent_id] = {
                    'loss_speed': 0.0,
                    'loss_toll': 0.0,
                    'alpha': 0.5
                }
            return losses
            # return dict(zip(self.agent_ids, [0.0 for _ in range(len(self.agent_ids))]))

        transitions, indices, weights = self.replay_memory.sample(self.batch_size)

        for agent_id in self.agent_ids:
            batch = Transition(*zip(*transitions[agent_id]))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_obs)),
                                          dtype=torch.bool)  # 是否终结（True, True, ..., False）
            non_final_next_obs = [s for s in batch.next_obs if s is not None]
            obs_batch = list(batch.obs)
            action_batch = torch.cat(batch.action)
            # 对 reward 进行拆分：分别获取 speed 与 toll 的奖励
            reward_speed_batch = torch.cat([r["speed"] for r in batch.reward])
            reward_toll_batch = torch.cat([r["toll"] for r in batch.reward])
            # print(batch.reward)
            # print(reward_toll_batch)

            # 计算当前 Q 值
            q_speed, q_toll, _ = self.q_networks[agent_id](obs_batch)
            q_speed = q_speed.gather(1, action_batch)
            q_toll = q_toll.gather(1, action_batch)

            next_obs_value_speed = torch.zeros(self.batch_size, device=self.device)
            next_obs_value_toll = torch.zeros(self.batch_size, device=self.device)
            if non_final_next_obs:
                with torch.no_grad():
                    next_q_speed, next_q_toll, _ = self.target_networks[agent_id](non_final_next_obs)
                    max_next_q_speed = next_q_speed.max(1)[0]
                    max_next_q_toll = next_q_toll.max(1)[0]
                    next_obs_value_speed[non_final_mask] = max_next_q_speed.detach()
                    next_obs_value_toll[non_final_mask] = max_next_q_toll.detach()
            expected_q_speed = reward_speed_batch + (self.gamma * torch.unsqueeze(next_obs_value_speed, 1))
            expected_q_toll = reward_toll_batch + (self.gamma * torch.unsqueeze(next_obs_value_toll, 1))
            diff_speed = expected_q_speed - q_speed
            diff_toll = expected_q_toll - q_toll

            # 为了消除量纲，对 diff 分别归一化
            std_speed = torch.std(diff_speed) + 1e-5
            std_toll = torch.std(diff_toll) + 1e-5

            # ---- Pareto-based priority update ----
            td_errors = torch.cat([
                diff_speed.detach().squeeze().abs().cpu().unsqueeze(1),
                diff_toll.detach().squeeze().abs().cpu().unsqueeze(1)
            ], dim=1).numpy()  # shape: [batch_size, 2]

            N = td_errors.shape[0]
            priorities = np.zeros(N)

            for i in range(N):
                dominated = False
                for j in range(N):
                    if i != j:
                        if (td_errors[j][0] >= td_errors[i][0] and
                                td_errors[j][1] >= td_errors[i][1] and
                                (td_errors[j][0] > td_errors[i][0] or
                                 td_errors[j][1] > td_errors[i][1])):
                            dominated = True
                            break
                if not dominated:
                    priorities[i] = np.sum(td_errors[i])  # 前沿点：用两个误差的和
                else:
                    priorities[i] = 1e-5  # 被支配点：给极小值
            self.replay_memory.update_priority(agent_id, indices[agent_id], priorities.tolist())

            loss_speed = self.criterion(q_speed, expected_q_speed) * weights[agent_id]
            loss_speed = loss_speed.mean()
            self.speed_running_mean = self.beta * self.speed_running_mean + (1 - self.beta) * loss_speed.item()
            self.speed_running_var = self.beta * self.speed_running_var + (1 - self.beta) * (
                    loss_speed.item() - self.speed_running_mean) ** 2
            loss_speed_norm = loss_speed / (
                    torch.sqrt(torch.tensor(self.speed_running_var)) + self.eps)
            loss_toll = self.criterion(q_toll, expected_q_toll) * weights[agent_id]
            loss_toll = loss_toll.mean()
            self.toll_running_mean = self.beta * self.toll_running_mean + (1 - self.beta) * loss_toll.item()
            self.toll_running_var = self.beta * self.toll_running_var + (1 - self.beta) * (
                    loss_toll.item() - self.toll_running_mean) ** 2
            loss_toll_norm = loss_toll / (
                    torch.sqrt(torch.tensor(self.toll_running_var)) + self.eps)
            loss_total = loss_speed_norm * 0.5 + loss_toll_norm * 0.5
            self.optims[agent_id].zero_grad()
            loss_total.backward()
            self.optims[agent_id].step()
            losses[agent_id] = {"loss_speed": loss_speed.item(), "loss_toll": loss_toll.item(),
                                "alpha": None}
        return losses

    def get_grad_vector(self, loss, model, shared_params):
        grads = torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=True, allow_unused=True)
        grad_vector = torch.cat([g.view(-1) for g in grads if g is not None])
        return grad_vector

    def compute_pareto_weight(self, g_speed, g_toll):
        g_diff = g_speed - g_toll
        dot_val = torch.dot(g_toll, g_diff)
        norm_sq = torch.dot(g_diff, g_diff) + 1e-8
        alpha = -dot_val / norm_sq
        alpha = alpha.clamp(0, 1)
        return alpha

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
        return {"reward_speed": [], "reward_toll": [], "loss": [], "horizon_length": [], "ttt": [], "ttr": []}

    def training_record(self, res, episode_records, env):
        mean_reward_speed = 0
        mean_reward_toll = 0
        mean_loss = 0
        for agent_id in self.agent_ids:
            for i in range(len(episode_records["loss"])):
                mean_reward_speed += episode_records["reward"][i][agent_id]["speed"]
                mean_reward_toll += episode_records["reward"][i][agent_id]["toll"]
                mean_loss += episode_records["loss"][i][agent_id]["loss_speed"] + episode_records["loss"][i][agent_id][
                    "loss_toll"]
        n = len(self.agent_ids) * len(episode_records["loss"])
        # mean_reward_speed = (mean_reward_speed / len(self.agent_ids)).detach().cpu().numpy()[0, 0]
        mean_reward_speed = (mean_reward_speed / n).detach().cpu().numpy()[0, 0]
        # print("reward_speed:",mean_reward_speed)
        mean_reward_toll = \
            (mean_reward_toll * (self.controllable_agent_num / len(self.agent_ids))).detach().cpu().numpy()[0, 0]
        mean_reward_toll /= 1000
        mean_loss = mean_loss / n
        res["reward_speed"].append(mean_reward_speed)
        res["reward_toll"].append(mean_reward_toll)
        res["loss"].append(mean_loss)
        res["horizon_length"].append(len(episode_records["obs"]))
        res["ttt"].append(env.get_episode_reward_time())
        res["ttr"].append(env.get_episode_reward_toll())
        return res



    def get_pareto_front_indices(self, points):
        num_points = points.shape[0]
        is_dominated = np.zeros(num_points, dtype=bool)
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                        is_dominated[i] = True
                        break
        return np.where(~is_dominated)[0]

    # 计算 Pareto 前沿的动作索引
    def pareto_front_indices(self, q1, q2):
        indices = []
        num = len(q1)
        for i in range(num):
            dominated = False
            for j in range(num):
                if i != j:
                    # 如果动作 j 在两个目标上均不低于动作 i，并且至少在一个目标上严格更好，则动作 i 被支配
                    if q1[j] >= q1[i] and q2[j] >= q2[i] and (q1[j] > q1[i] or q2[j] > q2[i]):
                        dominated = True
                        break
            if not dominated:
                indices.append(i)
        return indices

    def save_results(self, training_record_dict, savepath):
        training_record_dict = pd.DataFrame(training_record_dict)
        training_record_dict.to_csv(savepath, index=False)
