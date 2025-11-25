import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from controls.envs.metagraph import GraphData

class CommGraph(object):
    def __init__(self, base_env, meta_graph, device, embedding_dims, random_std=0):
        self.base_env = base_env
        self.meta_graph = meta_graph
        self.device = device
        self.embedding_dims = embedding_dims
        self.random_std = random_std

        self.agent_list = self.meta_graph.controllable_agent_indices
        self.pos = dict(zip(
            range(len(self.agent_list)),
            [self.meta_graph.pos[each] for each in self.agent_list]))

        self.edge_list = []
        self.data_networkx = None
        self.data_pyg = None

    def generate_edge_list(self):
        for i in range(len(self.agent_list)):
            for j in range(i + 1, len(self.agent_list)):
                self.edge_list.append((i, j))

    def build_networkx(self):
        G = nx.Graph()
        G.add_edges_from(self.edge_list)
        self.data_networkx = G

    def build_data_pyg(self):
        x = np.zeros([len(self.agent_list), self.embedding_dims], dtype=np.float32)
        edge_index = np.array(list(zip(*self.edge_list)), dtype=np.int64)
        y = np.zeros(len(self.agent_list), dtype=np.float32)
        batch = np.zeros(len(self.agent_list), dtype=np.float32)
        pos = np.array([self.meta_graph.pos[each] for each in self.agent_list], dtype=np.float32)
        self.data_pyg = GraphData.from_data(x=x, edge_index=edge_index, y=y, batch=batch, pos=pos)

    def build_graph(self):
        self.generate_edge_list()
        self.build_data_pyg()
        self.build_networkx()

    def show_graph(self):
        nx.draw(self.data_networkx, pos=self.pos, with_labels=True)
        plt.show()

    def compute_embeddings(self, x):
        out = None
        return None


if __name__ == "__main__":
    import monstac_api
    from pathlib import Path
    import os
    from controls.envs.metagraph import MetaGraph
    p = Path(__file__).absolute()
    pparts = list(p.parts)
    count = -1
    for i, each in enumerate(pparts):
        if each == "Monstac":
            count = i
            break
    rootpath = os.path.join(*pparts[:count + 1])
    os.chdir(rootpath)

    scen_path = os.path.join(rootpath, "scenarios", "ToyNetwork_perimeter.json")

    base_env = monstac_api.general.Monstac(scen_path)
    a = MetaGraph(base_env, 100, 0)
    a.build_graph()
    b = CommGraph(base_env, a, 100, 0)
    b.build_graph()
    b.show_graph()