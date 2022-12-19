from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def row_layout(*args):
    shape = list(args[0].shape); shape[0] *= 2
    return torch.cat([a.unsqueeze(dim = 1) for a in args], dim = 1).reshape(shape)

COORD_LIM = 100

def tsp_gen(batch_size, n_nodes):
    x = torch.randint(COORD_LIM, (batch_size, n_nodes, 2)).to(device)
    adj = torch.cdist(x.float(), x.float(), p = 2)
    return x, adj

def tsp_plot(x, y, sol, layout):
    batch_size, n_nodes, _ = x.shape
    assert layout[0] * layout[1] == batch_size
    fig = plt.figure(figsize = (4 * layout[1], 4 * layout[0]))
    for i in range(batch_size):
        ax = fig.add_subplot(layout[0], layout[1], i + 1)
        plt.scatter(x[i, :, 0].cpu().data.numpy(), x[i, :, 1].cpu().data.numpy(), s = 5)
        lines = [[x[i, sol[i, j]].tolist(), x[i, sol[i, (j + 1) % n_nodes]].tolist()] for j in range(n_nodes)]
        ax.add_collection(mc.LineCollection(lines, linewidths = 1))
        ax.autoscale()
        ax.margins(0.1)
        ax.set_title(y[i].item())
    return plt.show()

def tsp_cc(x, adj, verbose = False): # concorde
    batch_size, n_nodes, _ = x.shape
    y = []
    tours = []
    tbar = range(batch_size)
    if verbose:
        tbar = tqdm(tbar)
    for i in tbar:
        solver = TSPSolver.from_data(x[i, :, 0].cpu().data.numpy(), x[i, :, 1].cpu().data.numpy(), norm = 'EUC_2D')
        sol = solver.solve()
        assert sol.found_tour
        tours.append(sol.tour)
        y.append(sum([adj[i, sol.tour[-1], sol.tour[0]].item(), *[adj[i, sol.tour[j], sol.tour[j + 1]].item() for j in range(len(sol.tour) - 1)]]))
    return torch.FloatTensor(y).to(device), torch.LongTensor(np.stack(tours, axis = 0)).to(device)

def tsp_sample(adj, ze, mode = 'softmax', samples = 1, epsilon = 0.): # epsilon exploration
    assert mode in ['softmax', 'greedy']
    if mode == 'greedy':
        assert samples == 1
    batch_size, n_nodes, _ = adj.shape
    zex = ze.expand((samples, batch_size, n_nodes, n_nodes))
    adj_flat = adj.view(batch_size, n_nodes * n_nodes).expand((samples, batch_size, n_nodes * n_nodes))
    idx = torch.arange(n_nodes).expand((samples, batch_size, n_nodes)).to(device)
    mask = torch.ones((samples, batch_size, n_nodes), dtype = torch.bool).to(device)
    maskFalse = torch.zeros((samples, batch_size, 1), dtype = torch.bool).to(device)
    v0 = u = torch.zeros((samples, batch_size, 1), dtype = torch.long).to(device) # starts from v0:=0
    mask.scatter_(dim = -1, index = u, src = maskFalse)
    y = []
    if mode == 'softmax':
        logp, logq = [], []
    else:
        sol = [u]
    for i in range(1, n_nodes):
        zei = zex.gather(dim = -2, index = u.unsqueeze(dim = -1).expand((samples, batch_size, 1, n_nodes))).squeeze(dim = -2).masked_select(mask.clone()).view(samples, batch_size, n_nodes - i)
        if mode == 'softmax':
            pei = F.softmax(zei, dim = -1)
            qei = epsilon / (n_nodes - i) + (1. - epsilon) * pei
            vi = qei.view(samples * batch_size, n_nodes - i).multinomial(num_samples = 1, replacement = True).view(samples, batch_size, 1)
            logp.append(torch.log(pei.gather(dim = -1, index = vi)))
            logq.append(torch.log(qei.gather(dim = -1, index = vi)))
        elif mode == 'greedy':
            vi = zei.argmax(dim = -1, keepdim = True)
        v = idx.masked_select(mask).view(samples, batch_size, n_nodes - i).gather(dim = -1, index = vi)
        y.append(adj_flat.gather(dim = -1, index = u * n_nodes + v))
        u = v
        mask.scatter_(dim = -1, index = u, src = maskFalse)
        if mode == 'greedy':
            sol.append(u)
    y.append(adj_flat.gather(dim = -1, index = u * n_nodes + v0)) # ends at node v0
    y = torch.cat(y, dim = -1).sum(dim = -1).T # (batch_size, samples)
    if mode == 'softmax':
        logp = torch.cat(logp, dim = -1).sum(dim = -1).T
        logq = torch.cat(logq, dim = -1).sum(dim = -1).T
        return y, logp, logq # (batch_size, samples)
    elif mode == 'greedy':
        return y.squeeze(dim = 1), torch.cat(sol, dim = -1).squeeze(dim = 0) # (batch_size,)

def tsp_greedy(adj, ze):
    return tsp_sample(adj, ze, mode = 'greedy') # y, sol

def tsp_optim(adj, ze0, opt_fn, steps, samples, epsilon = 0., show = 0, show_freq = 100, verbose = False, x = None):
    batch_size, n_nodes, _ = adj.shape
    ze = nn.Parameter(ze0.to(device), requires_grad = True)
    opt = opt_fn([ze])
    y_means = []
    tbar = range(1, steps + 1)
    if verbose:
        tbar = tqdm(tbar)
    y_bl = torch.zeros((batch_size, 1)).to(device)
    if show > 0:
        sol = [tsp_greedy(adj[ : show], ze[ : show])]
    for t in tbar:
        opt.zero_grad()
        y, logp, logq = tsp_sample(adj, ze, 'softmax', samples, epsilon)
        y_means.append(y.mean().item())
        if verbose:
            tbar.set_description(f'step={t} y_mean={y_means[-1]:.4f}')
        y_bl = y.mean(dim = -1, keepdim = True)
        J = (((y - y_bl) * torch.exp(logp - logq)).detach() * logp).mean(dim = -1).sum()
        J.backward()
        opt.step()
        if show > 0 and t % show_freq == 0:
            sol.append(tsp_greedy(adj[ : show], ze[ : show]))
    if verbose:
        ts = np.arange(1, steps + 1)
        sns.lineplot(x = ts, y = y_means)
        plt.title('E[y] vs step')
        plt.show()
    if show > 0:
        assert x is not None
        tsp_plot(
            x[ : show].repeat(len(sol), 1, 1),
            torch.cat([s[0] for s in sol], dim = 0),
            torch.cat([s[1] for s in sol], dim = 0),
            (len(sol), show),
        )
    return ze

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch_geometric.nn as gnn

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_nodes = args.n_nodes
        self.depth = args.depth
        self.units = args.units
        self.v_lin0 = nn.Linear(2, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin1 = nn.Linear(self.units, 1)
        self.act = args.act
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.edge_index = torch.LongTensor([
            [i * self.n_nodes + u for i in range(self.batch_size) for u in range(self.n_nodes) for v in range(self.n_nodes) if u != v],
            [i * self.n_nodes + v for i in range(self.batch_size) for u in range(self.n_nodes) for v in range(self.n_nodes) if u != v],
        ]).to(device)
        self.e_mask = torch.BoolTensor(
            [[u != v for v in range(self.n_nodes)] for u in range(self.n_nodes)]
        ).expand(self.batch_size, self.n_nodes, self.n_nodes).to(device)
        self.par0 = torch.zeros((self.batch_size, self.n_nodes, self.n_nodes)).to(device)
    def forward(self, x, adj):
        #print('B=', self.batch_size, x.size(0), adj.size(0))
        #print('n=', self.n_nodes, x.size(1), adj.size(1), adj.size(2))
        #print('h=', x.size(2))
        x = x / COORD_LIM
        adj = adj / COORD_LIM
        x = x.view(self.batch_size * self.n_nodes, 2)
        w = adj.masked_select(self.e_mask).unsqueeze(dim = -1)
        x = self.v_lin0(x)
        x = self.act(x)
        w = self.e_lin0(w)
        w = self.act(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act(self.v_bns[i](x1 + gnn.global_max_pool(w2 * x2[self.edge_index[1]], self.edge_index[0])))
            w = w0 + self.act(self.e_bns[i](w1 + x3[self.edge_index[0]] + x4[self.edge_index[1]]))
        w = self.e_lin1(w)
        par0 = self.par0.masked_scatter(self.e_mask, w.view(-1))
        return par0

NET_SAVE_PATH = 'net%d.pt'

def net_train(args, net, verbose = False):
    net.train()
    net.set_batch_size(args.tr_batch_size)
    opt = args.opt_outer_fn(net.parameters())
    tbar = range(1, args.tr_outer_epochs + 1)
    if verbose:
        tbar = tqdm(tbar)
        losses = []
    for epoch in tbar:
        opt.zero_grad()
        x, adj = tsp_gen(args.tr_batch_size, args.n_nodes)
        par0 = net(x, adj)
        par1 = tsp_optim(adj, par0, args.opt_inner_fn, args.tr_inner_epochs, args.tr_inner_samples)
        par0.backward(par1.grad / args.tr_batch_size)
        opt.step()
        if verbose:
            losses.append(tsp_greedy(adj, par1)[0].mean().item())
            tbar.set_description(f'[epoch {epoch}] loss={losses[-1]:.4f}')
        net.eval()
        torch.save(net.state_dict(), NET_SAVE_PATH % epoch)
        net.train()
    if verbose:
        ts = np.arange(1, args.tr_outer_epochs + 1)
        sns.lineplot(x = ts, y = losses)
        plt.title('loss vs epoch')
        plt.show()
    return net

args = Dict(
    n_nodes = 100,
    opt_outer_fn = lambda par: optim.AdamW(par, lr = 1e-3, weight_decay = 5e-4),
    opt_inner_fn = lambda par: optim.AdamW(par, lr = 1e-1, weight_decay = 0.),
    tr_batch_size = 64,
    tr_outer_epochs = 100,
    tr_inner_epochs = 100,
    tr_inner_samples = 100,
    te_batch_size = 20,
    te_outer_epochs = 1,
    te_inner_epochs = 2000,
    te_inner_samples = 1000,
    act = F.leaky_relu,
    units = 64,
    depth = 3,
)

net = Net(args).to(device)
net = net_train(args, net, verbose = True)