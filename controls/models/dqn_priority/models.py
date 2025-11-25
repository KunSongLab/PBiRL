# import torch
# import torch_geometric
# from torch.nn import Linear, ReLU
# from torch_geometric.nn import GATConv, global_mean_pool
#
# class DQNModel(torch.nn.Module):
#     def __init__(self, input_dims, hidden_dims, output_dims):
#         super(DQNModel, self).__init__()
#         self.gat_conv = GATConv(
#             input_dims,
#             hidden_dims,
#             heads=1,
#             edge_dim=3
#         )
#         # self.relu = ReLU()
#         self.pooling = global_mean_pool
#         self.output_layer = Linear(hidden_dims, output_dims)
#
#     def forward(self, input_data):
#         outs = []
#         for batch_data in input_data:
#             x = batch_data["own"]["x"]
#             edge_index = batch_data["own"]["edge_index"]
#             edge_attr = batch_data["own"]["edge_attr"]
#             batch = batch_data["own"]["batch"]
#             out = self.gat_conv(x, edge_index, edge_attr)
#             out = self.pooling(out, batch)
#             outs.append(out)
#         return self.output_layer(torch.concat(outs, 0))

import torch

class DQNModel(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(DQNModel, self).__init__()
        self.input_dims = input_dims
        self.dqn_network = torch.nn.Sequential(
            torch.nn.Linear(input_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, input_data):
        xs = []
        for batch_data in input_data:
            x = batch_data["own"]["x"][:, 4:].flatten()
            edge_attr = batch_data["own"]["edge_attr"][:, 1].flatten()
            # xs.append(x)
            xs.append(torch.cat([x, edge_attr]))
            # xs.append(torch.from_numpy(batch_data))
        return self.dqn_network(torch.stack(xs))