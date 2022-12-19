from inc.header import *
from inc.utils import *

__TSP_VERSION__ = 1.1

# generate save_name
def tsp_save_name(args, save_name = None):
    if not save_name:
        timestamp = int(time.time() * 100)
        save_name = f'dimes{__TSP_VERSION__}-tsp{num_abbr(args.n_nodes)}-knn{args.knn_k}@{timestamp}'
    return osp.join(args.output_dir, save_name)

# interfaces for torch_sampling

@torch.no_grad()
def tsp_greedy(graph, par, sample_size, best = True):
    ys, tours = pysa.tsp_greedy(graph.x, graph.degs, graph.edge_index[1], par, sample_size)
    if best:
        y, i = ys.min(dim = 0)
        return y.detach().clone(), tours[i].detach().clone()
    else:
        return ys.detach().clone(), tours.detach().clone()

@torch.no_grad()
def tsp_softmax(graph, par, sample_size, y_bl = None, best = True):
    ys, tours = pysa.tsp_softmax(graph.x, graph.degs, graph.edge_index[1], par, sample_size, np.nan if y_bl is None else y_bl)
    if best:
        y, i = ys.min(dim = 0)
        return y.detach().clone(), tours[i].detach().clone()
    else:
        return ys.detach().clone(), tours.detach().clone()

@torch.no_grad()
def tsp_softmax_grad(graph, par, sample_size, y_bl = None):
    ys, grad = pysa.tsp_softmax_grad(graph.x, graph.degs, graph.edge_index[1], par, sample_size, np.nan if y_bl is None else y_bl)
    return ys.detach().clone(), grad.detach().clone()

# calculate cost from coordinates
def tsp_calc_cost_x(sol, x, p = 2.):
    sol_sizes = sol.size()
    sol = torch.cat([sol, sol[:, 0].unsqueeze(1)], dim = 1)
    return F.pairwise_distance(
        x.gather(dim = 1, index = sol[:, : -1].unsqueeze(2).expand(*sol_sizes, 2)).flatten(0, 1),
        x.gather(dim = 1, index = sol[:, 1 : ].unsqueeze(2).expand(*sol_sizes, 2)).flatten(0, 1),
        p = p).view(*sol_sizes).sum(dim = 1)

# calculate cost from distance matrix
def tsp_calc_cost_adj(sol, adj):
    batch_size, n_nodes = sol.size()
    sol = torch.cat([sol, sol[:, 0].unsqueeze(1)], dim = 1)
    bi = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_nodes).flatten()
    return adj[bi, sol[:, : -1].flatten(), sol[:, 1 : ].flatten()].view(batch_size, n_nodes).sum(dim = 1)

# convert theta to heatmap
def tsp_make_heatmap(n_nodes, par, edge_index, fname):
    idx = torch.argsort(edge_index[0])
    par = par[idx]
    edge_index = edge_index[:, idx]
    cnt = torch.unique_consecutive(edge_index[0], return_inverse = False, return_counts = True)[1].tolist()
    par = torch.split(par, cnt)
    edge_index = torch.split(edge_index, cnt, dim = 1)
    assert len(par) == len(edge_index)
    probs = torch.zeros((n_nodes, n_nodes), dtype = par[0].dtype, device = par[0].device)
    tbar = trange(len(par), desc = fname + ' computing')
    for i in tbar:
        probs[edge_index[i][0, 0], edge_index[i][1]] = F.softmax(par[i])
        if i == len(par) - 1:
            tbar.set_description(fname + ' saving')
    return np.savetxt(fname, probs.cpu().detach().numpy(), fmt = '%.6f', delimiter = ' ', header = f'{n_nodes}', comments = '')
