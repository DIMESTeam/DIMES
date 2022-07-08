TR_TIMESTAMP = 164751131945
TR_PATH = 'output/tsp'

from inc.tsp_args import *
from inc.tsp_core import *

args = args_init(**args_dict)
save_name = tsp_save_name(args, timestamp = TR_TIMESTAMP)

x_list = torch.tensor(np.load(f'input/testt-{args.n_nodes}-coords.npy'), dtype = torch.float32, device = args.device)
graph_list = [Graph.knn(x, args.knn_k) for x in tqdm(x_list[args.te_range_l : args.te_range_r])]

import numpy as np
import scipy.special as ssp

def get_matrix(n_nodes, par, edge_index, probs = False):
    if probs:
        matrix = torch.zeros((n_nodes, n_nodes), dtype = par.dtype, device = par.device)
        idx = torch.argsort(edge_index[0])
        par = par[idx]
        edge_index = edge_index[:, idx]
        cnt = torch.unique_consecutive(edge_index[0], return_inverse = False, return_counts = True)[1].tolist()
        par = torch.split(par, cnt)
        edge_index = torch.split(edge_index, cnt, dim = 1)
        assert len(par) == len(edge_index)
        tbar = trange(len(par), desc = 'computing probs')
        for i in tbar:
            matrix[edge_index[i][0, 0], edge_index[i][1]] = F.softmax(par[i], dim = -1)
    else:
        matrix = torch.full((n_nodes, n_nodes), -1e9, dtype = par.dtype, device = par.device)
        matrix[edge_index[0], edge_index[1]] = par
    return matrix.cpu().detach().numpy()

def save_heatmap(i, probs, fname = None):
    if fname is None:
        fname = f'heatmap/tsp{args.n_nodes}/heatmaptsp{args.n_nodes}_{i}.txt'
    return np.savetxt(fname, probs, fmt = '%.6f', delimiter = ' ', header = f'{args.n_nodes}', comments = '')

tbar = trange(args.te_range_l, args.te_range_r)
for i in tbar:
    save_name_i = f'{save_name}~graph{i}'
    graph = graph_list[i - args.te_range_l]
    par1 = torch.load(osp.join(TR_PATH, save_name) + f'~graph{i}~par1.pt', map_location = args.device)
    matrix = get_matrix(args.n_nodes, par1, graph.edge_index)
    sorted_vector = np.sort(matrix, axis=-1)[:, -5].reshape(-1, 1)
    matrix[(matrix - sorted_vector) < 0] -= 1e9
    orig_matrix = matrix
    start = 1.0
    minimum = 0.0
    while minimum < 1e-4:
        matrix = ssp.softmax(orig_matrix * start, axis = -1)
        minimum = matrix[matrix > 0].min()
        start *= 0.5
    save_heatmap(i, matrix)
