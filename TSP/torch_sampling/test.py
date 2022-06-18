import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)

import torch_geometric as pyg
import torch_geometric.nn as gnn

class Graph:
    def __init__(self, x, edge_index, edge_attr):
        self.n_nodes = x.size(0)
        self.n_edges = edge_index.size(-1)
        self.x = x
        self.edge_index, self.edge_attr = pyg.utils.sort_edge_index(edge_index, edge_attr, num_nodes = self.n_nodes)
        self.data = pyg.data.Data(x = self.x, edge_index = self.edge_index, edge_attr = self.edge_attr).to(device)
        self.degs = self.edge_index[0].unique_consecutive(return_counts = True)[-1]

def tsp_gen(n_nodes_min, n_nodes_max, knn_k, to_undirected = False):
    n_nodes = random.randint(n_nodes_min, n_nodes_max)
    x = torch.rand(n_nodes, 2, device = device)
    edge_index = gnn.knn_graph(x, knn_k, flow = 'target_to_source').to(device)
    edge_attr = F.pairwise_distance(x[edge_index[0]], x[edge_index[1]], keepdim = True)
    if to_undirected:
        edge_index, edge_attr = pyg.utils.to_undirected(edge_index, edge_attr)
    return Graph(x = x, edge_index = edge_index, edge_attr = edge_attr)

from torch_sampling import tsp_greedy, tsp_softmax_grad
graph = tsp_gen(10000, 10000, 50)
par = torch.zeros(graph.n_edges, device = device)
res_gr = tsp_greedy(graph.x, graph.degs, graph.edge_index[1], par, 1000)
print(res_gr[0].min().item())
