import torch


class DQNModel(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(DQNModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dims, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims)
        # 两个目标对应两个不同的输出头
        self.head_speed = torch.nn.Linear(hidden_dims, output_dims)  # 针对平均通行时间（目标1）
        self.head_toll = torch.nn.Linear(hidden_dims, output_dims)  # 针对污染物排放（目标2）

    def forward(self, input_data):
        xs = []
        for batch_data in input_data:
            x = batch_data["own"]["x"][:, 4:].flatten()
            edge_attr = batch_data["own"]["edge_attr"][:, 1].flatten()
            xs.append(torch.cat([x, edge_attr]))
        x = torch.relu(self.fc1(torch.stack(xs)))
        shared = torch.relu(self.fc2(x))
        q_speed = self.head_speed(shared)
        q_toll = self.head_toll(shared)
        return q_speed, q_toll, shared  # 返回共享表示供梯度组合使用
