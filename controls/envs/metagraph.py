import os
import torch
import monstac_api
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import torch_geometric.utils.convert
from torch_geometric.data import Data  # self create a data type


class GraphData(object):
    def __init__(self):
        self.x = None
        self.edge_index = None
        self.edge_attr = None
        self.y = None
        self.batch = None
        self.pos = None
        self.center_id = None

    @classmethod
    def from_pyg(cls, pyg, center_id=None):
        obj = cls()
        # obj.x = pyg.x.detach().clone().numpy()
        # obj.edge_index = pyg.edge_index.detach().clone().numpy()
        # obj.edge_attr = pyg.edge_attr.detach().clone().numpy()
        # obj.y = pyg.y.detach().clone().numpy()
        # obj.batch = pyg.batch.detach().clone().numpy()
        # obj.pos = pyg.pos.detach().clone().numpy()

        obj.x = pyg.x.detach().clone()
        obj.edge_index = pyg.edge_index.detach().clone()
        obj.edge_attr = pyg.edge_attr.detach().clone()
        obj.y = pyg.y.detach().clone()
        obj.batch = pyg.batch.detach().clone()
        obj.pos = pyg.pos.detach().clone()
        obj.center_id = center_id
        return obj

    @classmethod
    def from_data(cls, x=None, edge_index=None, edge_attr=None, y=None, batch=None, pos=None, center_id=None):
        obj = cls()
        obj.x = x
        obj.edge_index = edge_index
        obj.edge_attr = edge_attr
        obj.y = y
        obj.batch = batch
        obj.pos = pos
        obj.center_id = center_id
        return obj


class MetaGraph(object):
    def __init__(self, base_env, device, embedding_dims, random_std=0):
        self.base_env = base_env
        self.device = device
        self.embedding_dims = embedding_dims
        self.random_std = random_std

        self.agent_ids = []
        self.init_node_obs = monstac_api.rl.get_init_node_observations(self.base_env)
        self.init_edge_obs = monstac_api.rl.get_init_edge_observations(self.base_env)

        self.node_attr_name = ["type", "capacity", "jam_density", "max_speed", "accumulation", "source", "sink"]
        self.edge_attr_name = ["type", "capacity", "flow"]

        self.node_id_table = {}
        self.source_ids = []
        self.target_ids = []
        self.controllable_agent_indices = []
        self.data_pyg = None
        self.data_networkx = None
        self.pos = None
        self.adj_mat = None

        # subgraph
        self.subgraphs_pyg = []
        self.subgraphs_networkx = []
        self.subgraphs_pos = []
        self.subgraphs_node_tables = []
        self.subgraph_edge_index = []  # avoid change memory between cpu and gpu.
        self.subgraphs_center_id = []
        self.edge_table = {}

    def init_index_tables(self):
        for i, each in enumerate(self.init_node_obs.node_id):
            self.node_id_table[each] = i

        for each in self.init_edge_obs.source_id:
            self.source_ids.append(self.node_id_table[each])

        for each in self.init_edge_obs.target_id:
            self.target_ids.append(self.node_id_table[each])

        for i, each in enumerate(zip(self.source_ids, self.target_ids)):
            self.edge_table[each] = i

        for each in monstac_api.rl.get_controllable_agent_index(self.base_env):
            self.controllable_agent_indices.append(self.node_id_table[each])

    def build_data_pyg(self):
        x = torch.tensor(list(zip(
            self.init_node_obs.type,
            self.init_node_obs.capacity,
            self.init_node_obs.jam_density,
            self.init_node_obs.max_speed,
            np.clip(np.array(self.init_node_obs.accumulation) / np.array(self.init_node_obs.jam_density), 0, 1),
            self.init_node_obs.source,
            self.init_node_obs.sink
        )), dtype=torch.float, device=self.device)
        # print("x:", x[:,0])
        # print("x_shape:", x.shape)
        edge_index = torch.tensor([self.source_ids, self.target_ids], dtype=torch.long, device=self.device)

        edge_attr = torch.tensor(list(zip(
            self.init_edge_obs.type,
            self.init_edge_obs.capacity,
            self.init_edge_obs.flow
        )), dtype=torch.float, device=self.device)

        y = torch.zeros(len(self.init_node_obs.node_id), dtype=torch.float, device=self.device)
        batch = torch.zeros(len(self.init_node_obs.node_id), dtype=torch.long, device=self.device)
        pos = torch.tensor(self.init_node_obs.location, device=self.device)

        self.data_pyg = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch, pos=pos)
        self.adj_mat = torch_geometric.utils.to_dense_adj(edge_index)[0]

    def build_data_networkx(self):
        aug_node_attr_names = ["node_id"] + self.node_attr_name
        aug_edge_attr_names = ["edge_id", "source_id", "target_id"] + self.edge_attr_name

        node_attr_dict = [dict(zip(aug_node_attr_names, each)) for each in zip(
            self.init_node_obs.node_id,
            self.init_node_obs.type,
            self.init_node_obs.capacity,
            self.init_node_obs.jam_density,
            self.init_node_obs.max_speed,
            self.init_node_obs.accumulation,
            self.init_node_obs.source,
            self.init_node_obs.sink
        )]

        edge_attr_dict = [dict(zip(aug_edge_attr_names, each)) for each in zip(
            self.init_edge_obs.edge_id,
            self.init_edge_obs.source_id,
            self.init_edge_obs.target_id,
            self.init_edge_obs.type,
            self.init_edge_obs.capacity,
            self.init_edge_obs.flow
        )]

        G = nx.DiGraph()
        G.add_nodes_from(tuple(zip(range(len(node_attr_dict)), node_attr_dict)))
        G.add_edges_from(tuple(zip(self.source_ids, self.target_ids, edge_attr_dict)))
        self.data_networkx = G
        self.pos = dict(zip(range(len(node_attr_dict)), np.array(self.init_node_obs.location)))

    def build_graph(self):
        self.init_index_tables()
        self.build_data_pyg()
        self.build_data_networkx()
        self.generate_controllable_subgraphs()

        self.data_pyg = GraphData.from_pyg(self.data_pyg)
        for i in range(len(self.subgraphs_pyg)):
            self.subgraphs_pyg[i] = GraphData.from_pyg(self.subgraphs_pyg[i], self.subgraphs_center_id[i])

    def show_graph(self):
        nx.draw(self.data_networkx, pos=self.pos, with_labels=True)
        plt.show()

    def generate_controllable_subgraphs(self):
        def _get_neighborhoods(node_id, adj_mat):
            neighborhoods = set()
            for i in range(len(adj_mat[node_id])):
                if adj_mat[node_id][i]:
                    neighborhoods.add(i)
            for i in range(len(adj_mat[:, node_id])):
                if adj_mat[i][node_id]:
                    neighborhoods.add(i)

            neighborhoods.add(node_id)  # add the node itself.
            return list(neighborhoods)

        for each in self.controllable_agent_indices:
            subset = sorted(_get_neighborhoods(each, self.adj_mat))
            sub_edge_index, sub_edge_attr = torch_geometric.utils.subgraph(
                subset,
                self.data_pyg.edge_index,
                self.data_pyg.edge_attr,
                relabel_nodes=True)

            temp_node_table = {}
            for i, item in enumerate(subset):
                temp_node_table[i] = item
            self.subgraphs_node_tables.append(temp_node_table)

            sub_pyg = Data(x=self.data_pyg.x[subset], edge_index=sub_edge_index, edge_attr=sub_edge_attr,
                           y=self.data_pyg.y[subset], batch=self.data_pyg.batch[subset], pos=self.data_pyg.pos[subset])
            sub_networkx = torch_geometric.utils.to_networkx(sub_pyg)
            sub_pos = {}
            for i, item in enumerate(subset):
                sub_pos[i] = self.pos[item]
            self.subgraphs_pyg.append(sub_pyg)
            self.subgraphs_networkx.append(sub_networkx)
            self.subgraphs_pos.append(sub_pos)
            self.subgraph_edge_index.append(sub_edge_index.cpu().numpy().tolist())
            self.subgraphs_center_id.append(subset.index(each))
        # print("controllable_agent_indices:", self.controllable_agent_indices)
        # print("subgraphs_center_id:", self.subgraphs_center_id)
        # print("subgraphs_pyg:",len(self.subgraphs_pyg))

    def show_subgraphs(self):
        for i, each in enumerate(self.subgraphs_networkx):
            nx.draw(each, pos=self.subgraphs_pos[i], with_labels=True)
            plt.show()

    def update_obs(self, accumulation, source, sink, flow):
        # for speed consideration, the item has been temporary blocked.
        # for i in range(len(accumulation)):
        #     # self.data_pyg.x[i, 4] = torch.clip(accumulation[i] / self.data_pyg.x[i, 2], 0, 1)
        #     # self.data_pyg.x[i, 4] = accumulation[i]
        #     self.data_pyg.x[i, 5] = source[i]
        #     self.data_pyg.x[i, 6] = sink[i]
        # for i in range(len(flow)):
        #     self.data_pyg.edge_attr[i, -1] = flow[i]

        # update data in subgraphs
        for i in range(len(self.subgraphs_pyg)):
            for key, value in self.subgraphs_node_tables[i].items():
                # self.subgraphs_pyg[i].x[key, 4] = torch.clip(accumulation[value] / self.subgraphs_pyg[i].x[key, 2], 0, 1)
                self.subgraphs_pyg[i].x[key, 4] = accumulation[value]
                self.subgraphs_pyg[i].x[key, 5] = source[value]
                self.subgraphs_pyg[i].x[key, 6] = sink[value]

            for j, edge_id in enumerate(zip(
                    self.subgraph_edge_index[i][0],
                    self.subgraph_edge_index[i][1]
            )):
                self.subgraphs_pyg[i].edge_attr[j, -1] = flow[self.edge_table[(
                    self.subgraphs_node_tables[i][edge_id[0]],
                    self.subgraphs_node_tables[i][edge_id[1]],
                )]]

        pass


if __name__ == "__main__":
    import monstac_api

    p = Path(__file__).absolute()
    pparts = list(p.parts)
    count = -1
    for i, each in enumerate(pparts):
        if each == "Monstac":
            count = i
            break
    rootpath = os.path.join(*pparts[:count + 1])
    os.chdir(rootpath)

    # scen_path = os.path.join(rootpath, "scenarios", "Kowloon.json")
    scen_path = os.path.join(rootpath, "scenarios", "RingFreeway.json")
    device = torch.device("cpu")
    base_env = monstac_api.general.Monstac(scen_path, 0, 0, 0)
    a = MetaGraph(base_env, device, 100, 0)
    a.build_graph()
    # a.show_graph()
    # a.show_subgraphs()
