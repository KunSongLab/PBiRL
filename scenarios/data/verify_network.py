import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_nodes(filename):
    res = {}
    data = pd.read_csv(filename)
    for i in range(len(data)):
        res[data["osmid"][i]] = np.array([data["x"][i], data["y"][i]])
    return res

def load_edges(filename):
    res = []
    data = pd.read_csv(filename)
    for i in range(len(data)):
        res.append((data["origin"][i], data["destination"][i]))
    return res

def create_nx(edges, pos):
    # G = nx.Graph()
    G = nx.DiGraph()
    G.add_edges_from(edges)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()

def plot_mfd(a, b):
    x = np.arange(0, 1500, 10)
    y = 53.874 * np.exp(-0.077 * a) * np.exp(-x * b / 3.161e6)
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.plot(x, x * y)
    plt.show()

if __name__ == "__main__":
    # node_filename = "ToyNetwork_small2/nodes.csv"
    # edge_filename = "ToyNetwork_small2/edges.csv"
    node_filename = "RingFreeway/nodes.csv"
    edge_filename = "RingFreeway/edges.csv"
    pos = load_nodes(node_filename)
    edges = load_edges(edge_filename)
    create_nx(edges, pos)
    plot_mfd(10, 2500)
    plot_mfd(10, 5000)
    plot_mfd(10, 7500)