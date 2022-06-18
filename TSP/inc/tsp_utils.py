from inc.header import *
from inc.utils import *

__TSP_VERSION__ = 1.1

def tsp_save_name(args, timestamp = None):
    if timestamp is None:
        timestamp = int(time.time() * 100)
    return f'{args.output_dir}/dimes{__TSP_VERSION__}-tsp{num_abbr(args.n_nodes)}-knn{args.knn_k}@{timestamp}'

@torch.no_grad()
def tsp_greedy(graph, par, sample_size):
    ys, tours = pysa.tsp_greedy(graph.x, graph.degs, graph.edge_index[1], par, sample_size)
    y, i = ys.min(dim = 0)
    return y.clone(), tours[i].clone()

@torch.no_grad()
def tsp_softmax_grad(graph, par, sample_size, y_bl = None):
    ys, grad = pysa.tsp_softmax_grad(graph.x, graph.degs, graph.edge_index[1], par, sample_size, np.nan if y_bl is None else y_bl)
    return ys.clone(), grad.clone()

def tsp_calc_cost_x(sol, x, p = 2.):
    sol_sizes = sol.size()
    sol = torch.cat([sol, sol[:, 0].unsqueeze(1)], dim = 1)
    return F.pairwise_distance(
        x.gather(dim = 1, index = sol[:, : -1].unsqueeze(2).expand(*sol_sizes, 2)).flatten(0, 1),
        x.gather(dim = 1, index = sol[:, 1 : ].unsqueeze(2).expand(*sol_sizes, 2)).flatten(0, 1),
        p = p).view(*sol_sizes).sum(dim = 1)

def tsp_calc_cost_adj(sol, adj):
    batch_size, n_nodes = sol.size()
    sol = torch.cat([sol, sol[:, 0].unsqueeze(1)], dim = 1)
    bi = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_nodes).flatten()
    return adj[bi, sol[:, : -1].flatten(), sol[:, 1 : ].flatten()].view(batch_size, n_nodes).sum(dim = 1)

def tsp_make_heatmap(n_nodes, par, edge_index, fname):
    idx = torch.argsort(edge_index[0])
    par = par[idx]
    edge_index = edge_index[:, idx]
    _, cnt = torch.unique_consecutive(edge_index[0], return_inverse = False, return_counts = True)[1]
    par = torch.split(par, cnt)
    edge_index = torch.split(edge_index, cnt, dim = 1)
    assert len(par) == len(edge_index)
    probs = torch.zeros((n_nodes, n_nodes), dtype = par.dtype, device = par.device)
    tbar = trange(len(par), desc = fname + ' computing')
    for i in tbar:
        probs[edge_index[i][0, 0], edge_index[i][1]] = F.softmax(par[i])
        if i == len(par) - 1:
            tbar.set_description(fname + ' saving')
    return np.savetxt(fname, probs.cpu().detach().numpy(), fmt = '%.6f', delimiter = ' ', header = f'{n_nodes}', comments = '')
