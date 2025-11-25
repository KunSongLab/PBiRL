import torch
import random
from collections import namedtuple, deque
from controls.models.dqn_priority.data_structures import SumSegmentTree, MinSegmentTree, circle_queue

Transition = namedtuple('Transition',
                        ('obs', 'action', 'next_obs', 'reward'))


class MAPrioritizedReplayMemory(object):
    def __init__(self, agent_ids, device, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.agent_ids = agent_ids
        self.memories = {}
        self.device = device
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        it_capacity = 1
        while it_capacity < self.capacity:
            it_capacity *= 2
        self.it_sums = {}
        self.it_mins = {}
        self.frames = {}
        self.max_priority = {}

        for agent_id in self.agent_ids:
            self.memories[agent_id] = circle_queue(capacity=self.capacity)
            self.it_sums[agent_id] = SumSegmentTree(it_capacity)
            self.it_mins[agent_id] = MinSegmentTree(it_capacity)
            self.frames[agent_id] = 1
            self.max_priority[agent_id] = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args):
        """Save a transition for each agent"""
        for agent_id in self.agent_ids:
            temp_idx = self.memories[agent_id].tail
            self.memories[agent_id].push(Transition(*[each[agent_id] for each in args]))
            self.it_sums[agent_id][temp_idx] = self.max_priority[agent_id] ** self.alpha
            self.it_mins[agent_id][temp_idx] = self.max_priority[agent_id] ** self.alpha

    def sample_proportional_indices(self, batch_size):
        res = {}
        for agent_id in self.agent_ids:
            res[agent_id] = []
            for i in range(batch_size):
                mass = random.random() * self.it_sums[agent_id].sum(0, len(self.memories[agent_id]) - 1)
                idx = self.it_sums[agent_id].find_prefixsum_idx(mass)
                res[agent_id].append(idx)

        return res

    def encode_sample(self, idxes):
        res = {}
        for agent_id in self.agent_ids:
            res[agent_id] = [self.memories[agent_id][i] for i in idxes[agent_id]]
        return res

    def sample_rank_indices(self, batch_size):
        raise NotImplementedError

    def sample(self, batch_size):
        indices = self.sample_proportional_indices(batch_size)
        encode_samples = self.encode_sample(indices)
        agent_weights = {}
        #find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        for agent_id in self.agent_ids:
            weights = []
            p_min = self.it_mins[agent_id].min() / self.it_sums[agent_id].sum()
            beta = self.beta_by_frame(self.frames[agent_id])
            self.frames[agent_id] += 1

            # max_weight given to smallest probability
            max_weight = (p_min * len(self.memories[agent_id])) ** (-beta)
            it_sum = self.it_sums[agent_id].sum()

            for idx in indices[agent_id]:
                p_sample = self.it_sums[agent_id][idx] / it_sum
                weight = (p_sample * len(self.memories[agent_id])) **(-beta)
                weights.append(weight / max_weight)

            agent_weights[agent_id] = torch.tensor(weights, device=self.device, dtype=torch.float32)
        return encode_samples, indices, agent_weights

    def update_priorities(self, indices, priorities):
        for agent_id in self.agent_ids:
            assert len(indices[agent_id]) == len(priorities[agent_id])
            for idx, priority in zip(indices[agent_id], priorities[agent_id]):
                assert 0 <= idx < len(self.memories[agent_id])
                self.it_sums[agent_id][idx] = (priority[agent_id] + 1e-5) ** self.alpha
                self.it_mins[agent_id][idx] = (priority[agent_id] + 1e-5) ** self.alpha
                self.max_priority[agent_id] = max(self.max_priority[agent_id],
                                                  (priority[agent_id] + 1e-5))

    def update_priority(self, agent_id, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < len(self.memories[agent_id])
            self.it_sums[agent_id][idx] = (priority + 1e-5) ** self.alpha
            self.it_mins[agent_id][idx] = (priority + 1e-5) ** self.alpha
            self.max_priority[agent_id] = max(self.max_priority[agent_id],
                                              (priority + 1e-5))

    def __len__(self):
        return len(self.memories[self.agent_ids[0]])