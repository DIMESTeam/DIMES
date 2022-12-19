from inc.header import *

# assert with custom exception class
def assert_(cond, cls, *args, **kwargs):
    if not cond:
        raise cls(*args, **kwargs)

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# preprocess instance graphs
class Graph:
    def __init__(self, x, edge_index, edge_attr):
        self.n_nodes = x.size(0)
        self.n_edges = edge_index.size(-1)
        self.x = x
        self.edge_index, self.edge_attr = pyg.utils.sort_edge_index(edge_index, edge_attr, num_nodes = self.n_nodes)
    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = pyg.data.Data(x = self.x, edge_index = self.edge_index, edge_attr = self.edge_attr)
        return self._data
    @property
    def degs(self):
        if not hasattr(self, '_degs'):
            self._degs = self.edge_index[0].unique_consecutive(return_counts = True)[-1]
        return self._degs
    @classmethod
    def knn(cls, x, k = None, to_undirected = False):
        n_nodes = x.size(0)
        if k is None:
            edge_index = get_edge_index(n_nodes, x.device)
        else:
            edge_index = gnn.knn_graph(x, k, flow = 'target_to_source').to(x.device)
        edge_attr = F.pairwise_distance(x[edge_index[0]], x[edge_index[1]], keepdim = True)
        if to_undirected:
            edge_index, edge_attr = pyg.utils.to_undirected(edge_index, edge_attr)
        return cls(x = x, edge_index = edge_index, edge_attr = edge_attr)
    @classmethod
    def gen(cls, device, n_nodes_min, n_nodes_max, knn_k = None, to_undirected = False):
        n_nodes = random.randint(n_nodes_min, n_nodes_max)
        x = torch.rand(n_nodes, 2, device = device)
        return cls.knn(x = x, k = knn_k, to_undirected = to_undirected)
    @classmethod
    def gen_batch(cls, batch_size, *args, **kwargs):
        graph_list = [cls.gen(*args, **kwargs) for i in range(batch_size)]
        return graph_list, cls.to_pyg_batch(graph_list)
    @staticmethod
    def to_pyg_batch(graph_list):
        return pyg.data.Batch.from_data_list([graph.data for graph in graph_list])

# edge_index of complete graphs
class EICG:
    def __init__(self, device):
        self.device = device
        self.ei_dict = dict()
    def make(self, n_nodes):
        return torch.tensor([
            [u for u in range(n_nodes) for v in range(n_nodes) if u != v],
            [v for u in range(n_nodes) for v in range(n_nodes) if u != v]
        ], dtype = torch.long, device = self.device)
    def get(self, n_nodes):
        if n_nodes not in self.ei_dict:
            self.ei_dict[n_nodes] = self.make(n_nodes)
        return self.ei_dict[n_nodes]
    def fill_mat(self, n_nodes, values):
        edge_index = self.get(n_nodes)
        mat = torch.zeros((n_nodes, n_nodes), dtype = values.dtype, device = values.device)
        mat[edge_index[0], edge_index[1]] = values
        return mat

def torch_add_grad(x, grad):
    if x.grad is None:
        x.grad = grad
    else:
        x.grad.add_(grad)
    return x

# calculate distance matrix from coordinates
def x_to_adj(x, p = 2.):
    return torch.cdist(x.float(), x.float(), p = p)

# sparse matrix to dense matrix
def sp_to_matrix(n_nodes, par_sp):
    edge_index = get_edge_index(n_nodes, par_sp.device)
    par = torch.zeros((n_nodes, n_nodes), dtype = par_sp.dtype, device = par_sp.device)
    par[edge_index[0], edge_index[1]] = par_sp
    return par

def num_abbr(n):
    if n < 1000:
        return f'{n}'
    elif n % 1000 == 0:
        return f'{n // 1000}k'
    else:
        return f'{n / 1000}k'
