from inc.tsp_args import *
from inc.tsp_core import *

args = args_init()
save_name = args.save_name
print(save_name)

mcts_dir = 'MCTS_500_1000' if args.n_nodes <= 1000 else 'MCTS_10000'

# load net
net = Net(args).to(args.device)
net.load_state_dict(torch.load(f'{save_name}~net{args.te_net}.pt', map_location = args.device))

# load data
x_list = torch.tensor(np.load(f'../data/test-{args.n_nodes}-coords.npy'), dtype = torch.float32, device = args.device)
graph_list = [Graph.knn(x, args.knn_k) for x in tqdm(x_list[args.te_range_l : args.te_range_r])]


import scipy.special as ssp

def get_matrix(n_nodes, par, edge_index, probs = False):
    matrix = torch.zeros((n_nodes, n_nodes), dtype = par.dtype, device = par.device)
    if probs:
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
        matrix[edge_index[0], edge_index[1]] = par
    return matrix.cpu().detach().numpy()

def save_heatmap(i, probs, fname = None):
    if fname is None:
        fname = f'{mcts_dir}/heatmap/tsp{args.n_nodes}/heatmaptsp{args.n_nodes}_{i}.txt'
    return np.savetxt(fname, probs, fmt = '%.6f', delimiter = ' ', header = f'{args.n_nodes}', comments = '')

# run active search & make heatmaps for MCTS
for i in trange(args.te_range_l, args.te_range_r):
    save_name_i = f'{save_name}~graph{i}'
    graph = graph_list[i - args.te_range_l]
    _, _, par1, _, _ = net_infer_greedy(args, net, graph, verbose = True, plot = True, save_name = save_name_i)
    matrix = get_matrix(args.n_nodes, par1, graph.edge_index)
    sorted_vector = np.sort(matrix, axis=-1)[:, -5].reshape(-1, 1)
    matrix[(matrix - sorted_vector) < 0] -= 1e9
    orig_matrix = matrix
    start = 1.0
    minimum = 0.0
    while minimum < 1e-4: # adjust temperature
        matrix = ssp.softmax(orig_matrix * start, axis = -1)
        minimum = matrix[matrix > 0].min()
        start *= 0.5
    save_heatmap(i, matrix)
